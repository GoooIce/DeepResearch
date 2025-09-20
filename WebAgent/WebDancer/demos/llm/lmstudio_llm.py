import copy
import json
import logging
import os
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Literal, Union

import lmstudio
from lmstudio import LMStudioError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel, simulate_response_completion_with_chat
from qwen_agent.llm.schema import ASSISTANT, Message, FunctionCall
from qwen_agent.log import logger


@register_llm('lmstudio')
class TextChatAtLMStudio(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF'  # Default model
        cfg = cfg or {}

        # Get API host configuration
        api_host = cfg.get('api_host')
        api_host = api_host or cfg.get('base_url')
        api_host = api_host or cfg.get('model_server')
        api_host = (api_host or '').strip()

        # Initialize LMStudio client
        try:
            if api_host:
                self.client = lmstudio.Client(api_host=api_host)
            else:
                self.client = lmstudio.Client()  # Use default client
        except Exception as ex:
            logger.warning(f"Failed to initialize LMStudio client: {ex}")
            self.client = None

    def _get_llm(self):
        """Get LLM instance from LMStudio."""
        if self.client is None:
            raise ModelServiceError("LMStudio client not initialized")
        
        try:
            # Get LLM instance for the specified model
            return lmstudio.llm(self.model)
        except Exception as ex:
            raise ModelServiceError(f"Failed to get LLM for model {self.model}: {ex}")

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            llm = self._get_llm()
            
            # Convert messages to LMStudio format
            history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            # Prepare prediction config
            config = self._prepare_prediction_config(generate_cfg)
            
            # Use respond_stream for chat streaming
            stream = llm.respond_stream(history=history, config=config)
            
            if delta_stream:
                # For delta streaming, yield each fragment as it comes
                for fragment in stream:
                    if hasattr(fragment, 'content') and fragment.content:
                        yield [Message(role=ASSISTANT, content=fragment.content, reasoning_content='')]
                    
            else:
                # For full streaming, accumulate the full response
                full_response = ""
                for fragment in stream:
                    if hasattr(fragment, 'content') and fragment.content:
                        full_response += fragment.content
                        # Yield the accumulated response each time
                        yield [Message(role=ASSISTANT, content=full_response, reasoning_content='')]
                    
        except LMStudioError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            llm = self._get_llm()
            
            # Convert messages to LMStudio format
            history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            # Prepare prediction config
            config = self._prepare_prediction_config(generate_cfg)
            
            # Use respond for non-streaming chat
            result = llm.respond(history=history, config=config)
            
            # Extract content from result
            content = result.content if hasattr(result, 'content') else str(result)
            
            return [Message(role=ASSISTANT, content=content)]
            
        except LMStudioError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        # For now, fall back to regular chat for function calling
        # LMStudio SDK has `act` method for tools, but we'll use the existing pattern
        generate_cfg = copy.deepcopy(generate_cfg)
        for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
            if k in generate_cfg:
                del generate_cfg[k]
        messages = simulate_response_completion_with_chat(messages)
        return self._chat(messages, stream=stream, delta_stream=delta_stream, generate_cfg=generate_cfg)

    def _chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        """Convert Message objects to dictionaries for LMStudio."""
        messages = [msg.model_dump() for msg in messages]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages

    def _prepare_prediction_config(self, generate_cfg: dict) -> Optional[Dict]:
        """Convert OpenAI-style config to LMStudio config."""
        if not generate_cfg:
            return None
            
        config = {}
        
        # Map common parameters
        if 'temperature' in generate_cfg:
            config['temperature'] = generate_cfg['temperature']
        if 'top_p' in generate_cfg:
            config['top_p'] = generate_cfg['top_p']
        if 'max_tokens' in generate_cfg:
            config['max_tokens'] = generate_cfg['max_tokens']
        if 'presence_penalty' in generate_cfg:
            config['presence_penalty'] = generate_cfg['presence_penalty']
        if 'frequency_penalty' in generate_cfg:
            config['frequency_penalty'] = generate_cfg['frequency_penalty']
        if 'stop' in generate_cfg:
            config['stop'] = generate_cfg['stop']
            
        return config if config else None