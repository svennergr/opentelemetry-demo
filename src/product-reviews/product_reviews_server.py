#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


# Python
import atexit
import json
import os
import random
import uuid
from typing import Any
from concurrent import futures

# Pip
import grpc
from opentelemetry import trace, metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

# Local
import logging
import demo_pb2
import demo_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from database import fetch_product_reviews as db_fetch_product_reviews
from database import fetch_product_reviews_from_db, fetch_avg_product_review_score_from_db

from openfeature import api
from openfeature.contrib.provider.flagd import FlagdProvider

from metrics import (
    init_metrics
)

# OpenAI
from openai import OpenAI
from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from sigil_sdk import (
    Client as SigilClient,
    ClientConfig,
    GenerationExportConfig,
    AuthConfig,
    GenerationStart,
    ModelRef,
    user_text_message,
    assistant_text_message,
)

from google.protobuf.json_format import MessageToJson

SIGIL_AGENT_NAME = "product-reviews-assistant"

llm_host = None
llm_port = None
llm_mock_url = None
llm_base_url = None
llm_api_key = None
llm_model = None
product_agent = None
product_catalog_stub = None
sigil_client = None

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers related to a specific product. "
    "Use tools as needed to fetch the product reviews and product information. "
    "Keep the response brief with no more than 1-2 sentences. "
    "If you don't know the answer, just say you don't know."
)

@function_tool
def fetch_product_reviews(product_id: str) -> str:
    """Executes a SQL query to retrieve reviews for a particular product."""
    return db_fetch_product_reviews(product_id)


@function_tool
def fetch_product_info(product_id: str) -> str:
    """Retrieves information for a particular product."""
    try:
        product = product_catalog_stub.GetProduct(demo_pb2.GetProductRequest(id=product_id))
        logger.info(f"product_catalog_stub.GetProduct returned: '{product}'")
        json_str = MessageToJson(product)
        return json_str
    except Exception as e:
        return json.dumps({"error": str(e)})


def build_product_agent():
    model_client = AsyncOpenAI(base_url=f"{llm_base_url}", api_key=f"{llm_api_key}")
    return Agent(
        name="Product Assistant",
        instructions=SYSTEM_PROMPT,
        tools=[fetch_product_reviews, fetch_product_info],
        model=OpenAIChatCompletionsModel(model=llm_model, openai_client=model_client),
    )

class ProductReviewService(demo_pb2_grpc.ProductReviewServiceServicer):
    def GetProductReviews(self, request, context):
        logger.info(f"Receive GetProductReviews for product id:{request.product_id}")
        product_reviews = get_product_reviews(request.product_id)

        return product_reviews

    def GetAverageProductReviewScore(self, request, context):
        logger.info(f"Receive GetAverageProductReviewScore for product id:{request.product_id}")
        product_reviews = get_average_product_review_score(request.product_id)

        return product_reviews

    def AskProductAIAssistant(self, request, context):
        logger.info(f"Receive AskProductAIAssistant for product id:{request.product_id}, question: {request.question}")
        history = [{"role": item.role, "content": item.content} for item in request.history]
        ai_assistant_response = get_ai_assistant_response(request.product_id, request.question, history)

        return ai_assistant_response

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)

def get_product_reviews(request_product_id):

    with tracer.start_as_current_span("get_product_reviews") as span:

        span.set_attribute("app.product.id", request_product_id)

        product_reviews = demo_pb2.GetProductReviewsResponse()
        records = fetch_product_reviews_from_db(request_product_id)

        for row in records:
            logger.info(f"  username: {row[0]}, description: {row[1]}, score: {str(row[2])}")
            product_reviews.product_reviews.add(
                    username=row[0],
                    description=row[1],
                    score=str(row[2])
            )

        span.set_attribute("app.product_reviews.count", len(product_reviews.product_reviews))

        # Collect metrics for this service
        product_review_svc_metrics["app_product_review_counter"].add(len(product_reviews.product_reviews), {'product.id': request_product_id})

        return product_reviews

def get_average_product_review_score(request_product_id):

    with tracer.start_as_current_span("get_average_product_review_score") as span:

        span.set_attribute("app.product.id", request_product_id)

        product_review_score = demo_pb2.GetAverageProductReviewScoreResponse()
        avg_score = fetch_avg_product_review_score_from_db(request_product_id)
        product_review_score.average_score = avg_score

        span.set_attribute("app.product_reviews.average_score", avg_score)

        return product_review_score

def maybe_return_rate_limit_error(question: str, request_product_id: str, conversation_id: str, span):
    llm_rate_limit_error = check_feature_flag("llmRateLimitError")
    logger.info(f"llmRateLimitError feature flag: {llm_rate_limit_error}")
    if not llm_rate_limit_error:
        return None

    random_number = random.random()
    logger.info(f"Generated a random number: {str(random_number)}")
    if random_number >= 0.5:
        return None

    client = OpenAI(
        base_url=f"{llm_mock_url}",
        api_key=f"{llm_api_key}",
    )
    user_prompt = f"Answer the following question about product ID:{request_product_id}: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("Invoking mock LLM with model: astronomy-llm-rate-limit")
    with sigil_client.start_generation(GenerationStart(
        conversation_id=conversation_id,
        agent_name=SIGIL_AGENT_NAME,
        model=ModelRef(provider="openai", name="astronomy-llm-rate-limit"),
    )) as gen_rec:
        try:
            client.chat.completions.create(
                model="astronomy-llm-rate-limit",
                messages=messages,
            )
        except Exception as e:
            gen_rec.set_call_error(e)
            logger.error(f"Caught Exception: {e}")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, description=str(e)))
            return "The system is unable to process your response. Please try again later."

    return None


def get_ai_assistant_response(request_product_id, question, history: list[dict[str, Any]]):

    with tracer.start_as_current_span("get_ai_assistant_response") as span:

        ai_assistant_response = demo_pb2.AskProductAIAssistantResponse()
        conversation_id = f"product-review-{uuid.uuid4()}"

        span.set_attribute("app.product.id", request_product_id)
        span.set_attribute("app.product.question", question)
        span.set_attribute("gen_ai.conversation.id", conversation_id)

        rate_limit_message = maybe_return_rate_limit_error(question, request_product_id, conversation_id, span)
        if rate_limit_message is not None:
            ai_assistant_response.response = rate_limit_message
            return ai_assistant_response

        sanitized_history = []
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant", "system") and isinstance(content, str) and content:
                sanitized_history.append({"role": role, "content": content})

        llm_inaccurate_response = check_feature_flag("llmInaccurateResponse")
        logger.info(f"llmInaccurateResponse feature flag: {llm_inaccurate_response}")

        user_prompt = f"Answer the following question about product ID:{request_product_id}: {question}"
        if llm_inaccurate_response and request_product_id == "L9ECAV7KIM":
            logger.info(f"Requesting inaccurate response for product_id: {request_product_id}")
            user_prompt = (
                f"{user_prompt} For this response, intentionally include one inaccurate detail."
            )

        input_items = sanitized_history + [{"role": "user", "content": user_prompt}]
        logger.info(f"Invoking product agent with input: '{input_items}'")

        with sigil_client.start_generation(GenerationStart(
            conversation_id=conversation_id,
            agent_name=SIGIL_AGENT_NAME,
            model=ModelRef(provider="openai", name=llm_model),
        )) as gen_rec:
            try:
                result = Runner.run_sync(
                    product_agent,
                    input_items,
                    max_turns=10,
                    run_config=RunConfig(),
                )
                response_text = str(result.final_output)
                gen_rec.set_result(
                    input=[user_text_message(user_prompt)],
                    output=[assistant_text_message(response_text)],
                )
                ai_assistant_response.response = response_text
                logger.info(f"Returning an AI assistant response: '{response_text}'")
            except Exception as e:
                gen_rec.set_call_error(e)
                logger.error(f"Caught Exception during agent run: {e}")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, description=str(e)))
                ai_assistant_response.response = "The system is unable to process your response. Please try again later."

        # Collect metrics for this service
        product_review_svc_metrics["app_ai_assistant_counter"].add(1, {'product.id': request_product_id})

        return ai_assistant_response

def must_map_env(key: str):
    value = os.environ.get(key)
    if value is None:
        raise Exception(f'{key} environment variable must be set')
    return value

def check_feature_flag(flag_name: str):
    # Initialize OpenFeature
    client = api.get_client()
    return client.get_boolean_value(flag_name, False)

if __name__ == "__main__":
    service_name = must_map_env('OTEL_SERVICE_NAME')

    api.set_provider(FlagdProvider(host=os.environ.get('FLAGD_HOST', 'flagd'), port=os.environ.get('FLAGD_PORT', 8013)))
    set_tracing_disabled(True)
    OpenAIAgentsInstrumentor().instrument()

    # Initialize Traces and Metrics
    tracer = trace.get_tracer_provider().get_tracer(service_name)
    meter = metrics.get_meter_provider().get_meter(service_name)

    product_review_svc_metrics = init_metrics(meter)

    # Initialize Logs
    logger_provider = LoggerProvider(
        resource=Resource.create(
            {
                'service.name': service_name,
            }
        ),
    )
    set_logger_provider(logger_provider)
    log_exporter = OTLPLogExporter(insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    # Attach OTLP handler to logger
    logger = logging.getLogger('main')
    logger.addHandler(handler)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add class to gRPC server
    service = ProductReviewService()
    demo_pb2_grpc.add_ProductReviewServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    llm_host = must_map_env('LLM_HOST')
    llm_port = must_map_env('LLM_PORT')
    llm_mock_url = f"http://{llm_host}:{llm_port}/v1"
    llm_base_url = must_map_env('LLM_BASE_URL')
    llm_api_key = must_map_env('OPENAI_API_KEY')
    llm_model = must_map_env('LLM_MODEL')

    sigil_export_protocol = os.environ.get('SIGIL_EXPORT_PROTOCOL', 'none')
    sigil_export_endpoint = os.environ.get('SIGIL_EXPORT_ENDPOINT', '')
    sigil_tenant_id = os.environ.get('SIGIL_TENANT_ID', '')
    sigil_auth = AuthConfig(mode='tenant', tenant_id=sigil_tenant_id) if sigil_tenant_id else AuthConfig(mode='none')
    sigil_client = SigilClient(ClientConfig(
        generation_export=GenerationExportConfig(
            protocol=sigil_export_protocol,
            endpoint=sigil_export_endpoint,
            auth=sigil_auth,
        ),
    ))
    atexit.register(sigil_client.shutdown)

    catalog_addr = must_map_env('PRODUCT_CATALOG_ADDR')
    pc_channel = grpc.insecure_channel(catalog_addr)
    product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(pc_channel)
    product_agent = build_product_agent()

    # Start server
    port = must_map_env('PRODUCT_REVIEWS_PORT')
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f'Product reviews service started, listening on port {port}')
    server.wait_for_termination()
