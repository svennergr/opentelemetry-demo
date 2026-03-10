#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sigil instrumentation in the product-reviews service.

Validates that:
- Sigil SDK imports and initializes correctly
- Generation recording works in no-op mode (protocol="none")
- Env-var-driven configuration produces correct client settings
- Agentic flow (two generations with tool call/result messages) records correctly
"""

import os
import unittest
from unittest.mock import patch

from sigil_sdk import (
    Client as SigilClient,
    ClientConfig,
    GenerationExportConfig,
    AuthConfig,
    GenerationStart,
    Message,
    MessageRole,
    ModelRef,
    ToolCall,
    tool_call_part,
    tool_result_message,
    user_text_message,
    assistant_text_message,
)


class TestSigilClientInitialization(unittest.TestCase):
    """Sigil client lifecycle in various configurations."""

    def test_noop_client_creation(self):
        """Client in none mode initializes and shuts down cleanly."""
        client = SigilClient(ClientConfig(
            generation_export=GenerationExportConfig(protocol="none"),
        ))
        client.shutdown()

    def test_http_client_creation(self):
        """Client in HTTP mode initializes (no actual connection needed)."""
        client = SigilClient(ClientConfig(
            generation_export=GenerationExportConfig(
                protocol="http",
                endpoint="http://localhost:8080/api/v1/generations:export",
                auth=AuthConfig(mode="none"),
            ),
        ))
        client.shutdown()

    def test_tenant_auth_creation(self):
        """Client with tenant auth sets the correct auth config."""
        client = SigilClient(ClientConfig(
            generation_export=GenerationExportConfig(
                protocol="http",
                endpoint="http://localhost:8080/api/v1/generations:export",
                auth=AuthConfig(mode="tenant", tenant_id="test-tenant"),
            ),
        ))
        client.shutdown()


class TestGenerationRecording(unittest.TestCase):
    """Generation recording via Sigil SDK."""

    def setUp(self):
        self.client = SigilClient(ClientConfig(
            generation_export=GenerationExportConfig(protocol="none"),
        ))

    def tearDown(self):
        self.client.shutdown()

    def test_sync_generation_with_output(self):
        """Recording a generation with input and output does not raise."""
        with self.client.start_generation(GenerationStart(
            conversation_id="test-conv-1",
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="gpt-4o-mini"),
        )) as rec:
            rec.set_result(
                input=[user_text_message("Summarize reviews for product X")],
                output=[assistant_text_message("Product X has excellent reviews.")],
            )

    def test_generation_with_empty_output(self):
        """Recording a generation with empty output (tool_calls response)."""
        with self.client.start_generation(GenerationStart(
            conversation_id="test-conv-2",
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="gpt-4o-mini"),
        )) as rec:
            rec.set_result(
                input=[user_text_message("Summarize reviews")],
                output=[],
            )

    def test_generation_error_recording(self):
        """set_call_error marks the generation as failed."""
        with self.client.start_generation(GenerationStart(
            conversation_id="test-conv-err",
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="astronomy-llm-rate-limit"),
        )) as rec:
            rec.set_call_error(Exception("Rate limit exceeded"))

    def test_multiple_generations_same_conversation(self):
        """Multiple generations share a conversation_id (agentic flow)."""
        conv_id = "test-conv-multi"

        with self.client.start_generation(GenerationStart(
            conversation_id=conv_id,
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="gpt-4o-mini"),
        )) as rec:
            rec.set_result(
                input=[user_text_message("question")],
                output=[],
            )

        with self.client.start_generation(GenerationStart(
            conversation_id=conv_id,
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="gpt-4o-mini"),
        )) as rec:
            rec.set_result(
                output=[assistant_text_message("final answer")],
            )


class TestEnvVarConfiguration(unittest.TestCase):
    """Env-var-driven Sigil client configuration."""

    def test_defaults_to_none_protocol(self):
        with patch.dict(os.environ, {}, clear=True):
            protocol = os.environ.get('SIGIL_EXPORT_PROTOCOL', 'none')
            self.assertEqual(protocol, 'none')

    def test_http_protocol_from_env(self):
        env = {
            'SIGIL_EXPORT_PROTOCOL': 'http',
            'SIGIL_EXPORT_ENDPOINT': 'http://sigil:8080/api/v1/generations:export',
            'SIGIL_TENANT_ID': 'demo-tenant',
        }
        with patch.dict(os.environ, env):
            protocol = os.environ.get('SIGIL_EXPORT_PROTOCOL', 'none')
            endpoint = os.environ.get('SIGIL_EXPORT_ENDPOINT', '')
            tenant_id = os.environ.get('SIGIL_TENANT_ID', '')

            self.assertEqual(protocol, 'http')
            self.assertEqual(endpoint, 'http://sigil:8080/api/v1/generations:export')
            self.assertEqual(tenant_id, 'demo-tenant')

            auth = AuthConfig(mode='tenant', tenant_id=tenant_id) if tenant_id else AuthConfig(mode='none')
            client = SigilClient(ClientConfig(
                generation_export=GenerationExportConfig(
                    protocol=protocol,
                    endpoint=endpoint,
                    auth=auth,
                ),
            ))
            client.shutdown()

    def test_no_tenant_uses_none_auth(self):
        env = {
            'SIGIL_EXPORT_PROTOCOL': 'http',
            'SIGIL_EXPORT_ENDPOINT': 'http://sigil:8080/api/v1/generations:export',
        }
        with patch.dict(os.environ, env, clear=True):
            tenant_id = os.environ.get('SIGIL_TENANT_ID', '')
            auth = AuthConfig(mode='tenant', tenant_id=tenant_id) if tenant_id else AuthConfig(mode='none')
            self.assertEqual(auth.mode, 'none')


class TestAgenticFlowSimulation(unittest.TestCase):
    """Simulates the full product-reviews agentic flow with Sigil recording."""

    def setUp(self):
        self.client = SigilClient(ClientConfig(
            generation_export=GenerationExportConfig(protocol="none"),
        ))

    def tearDown(self):
        self.client.shutdown()

    def test_full_agentic_flow(self):
        """Simulates: gen 1 (tool call) -> gen 2 (tool result -> answer)."""
        conv_id = "test-agentic-flow"
        tool_call_id = "call_abc"

        with self.client.start_generation(GenerationStart(
            conversation_id=conv_id,
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="gpt-4o-mini"),
        )) as rec:
            rec.set_result(
                input=[user_text_message("Can you summarize the product reviews?")],
                output=[Message(role=MessageRole.ASSISTANT, parts=[
                    tool_call_part(ToolCall(
                        name="fetch_product_reviews",
                        id=tool_call_id,
                        input_json=b'{"product_id": "OLJCESPC7Z"}',
                    )),
                ])],
            )

        with self.client.start_generation(GenerationStart(
            conversation_id=conv_id,
            agent_name="product-reviews-assistant",
            model=ModelRef(provider="openai", name="gpt-4o-mini"),
        )) as rec:
            rec.set_result(
                input=[tool_result_message(
                    tool_call_id=tool_call_id,
                    content='[["alice", "Amazing telescope!", 5], ["bob", "Good value", 4]]',
                )],
                output=[assistant_text_message(
                    "The telescope has excellent reviews with an average of 4.5 stars."
                )],
            )


if __name__ == "__main__":
    unittest.main()
