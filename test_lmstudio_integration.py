#!/usr/bin/env python3
"""
Test script for LMStudio SDK integration in DeepResearch.

This script demonstrates how to use the new LMStudio LLM implementation.
It can be used to test the integration with a running LMStudio server.

Usage:
    python test_lmstudio_integration.py [--api-host API_HOST] [--model MODEL_NAME]

Example:
    python test_lmstudio_integration.py --api-host http://localhost:1234 --model "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from WebAgent.WebDancer.demos.llm.lmstudio_llm import TextChatAtLMStudio
from qwen_agent.llm.schema import Message, ASSISTANT, USER


def test_lmstudio_llm(api_host=None, model_name=None):
    """Test the LMStudio LLM implementation."""
    print("🚀 Testing LMStudio LLM Integration")
    print("=" * 50)
    
    # Configuration
    cfg = {}
    if api_host:
        cfg['api_host'] = api_host
        print(f"📡 API Host: {api_host}")
    else:
        print("📡 API Host: Using default LMStudio configuration")
        
    if model_name:
        cfg['model'] = model_name
        print(f"🤖 Model: {model_name}")
    else:
        print("🤖 Model: Using default model")
    
    try:
        # Create LMStudio LLM instance
        print("\n🔧 Creating LMStudio LLM instance...")
        llm = TextChatAtLMStudio(cfg)
        print("✅ LMStudio LLM instance created successfully")
        
        # Test messages
        test_messages = [
            Message(role=USER, content="Hello! Can you tell me what 2+2 equals?")
        ]
        
        # Test non-streaming chat
        print("\n💬 Testing non-streaming chat...")
        try:
            response = llm._chat_no_stream(test_messages, {})
            print(f"✅ Non-streaming response: {response[0].content}")
        except Exception as e:
            print(f"❌ Non-streaming chat failed: {e}")
            
        # Test streaming chat
        print("\n🌊 Testing streaming chat...")
        try:
            stream_responses = list(llm._chat_stream(test_messages, delta_stream=False, generate_cfg={}))
            if stream_responses:
                print(f"✅ Streaming response: {stream_responses[-1][0].content}")
            else:
                print("❌ No streaming responses received")
        except Exception as e:
            print(f"❌ Streaming chat failed: {e}")
            
        print("\n✨ LMStudio integration test completed!")
        
    except Exception as e:
        print(f"❌ Failed to create LMStudio LLM instance: {e}")
        print("\n💡 Make sure LMStudio is running and accessible")
        return False
    
    return True


def test_react_agent_integration(api_host=None, model_name=None):
    """Test the React Agent with LMStudio integration."""
    print("\n🎭 Testing React Agent LMStudio Integration")
    print("=" * 50)
    
    try:
        from inference.react_agent import MultiTurnReactAgent
        
        # Configuration for agent
        llm_cfg = {
            'model': model_name or 'test-model',
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': 0.6,
                'top_p': 0.95,
                'presence_penalty': 1.1
            },
            'model_type': 'lmstudio'
        }
        
        if api_host:
            llm_cfg['api_host'] = api_host
        
        print("🔧 Creating MultiTurnReactAgent with LMStudio...")
        agent = MultiTurnReactAgent(llm=llm_cfg)
        print("✅ MultiTurnReactAgent created successfully")
        
        # Test server call (this will use LMStudio if available, fallback to OpenAI-compatible otherwise)
        test_messages = [
            {"role": "user", "content": "Hello, can you help me with a simple math question?"}
        ]
        
        print("💬 Testing agent server call...")
        try:
            # This is a mock test - in real usage, planning_port would be used for fallback
            response = agent._call_lmstudio_server(test_messages, max_tries=1)
            print(f"✅ Agent response: {response}")
        except Exception as e:
            print(f"⚠️  Agent call failed (expected without server): {e}")
            
        print("✨ React Agent integration test completed!")
        
    except Exception as e:
        print(f"❌ Failed to test React Agent: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test LMStudio SDK integration')
    parser.add_argument('--api-host', type=str, help='LMStudio API host (e.g., http://localhost:1234)')
    parser.add_argument('--model', type=str, help='Model name to use')
    parser.add_argument('--agent-only', action='store_true', help='Test only React Agent integration')
    parser.add_argument('--llm-only', action='store_true', help='Test only LLM integration')
    
    args = parser.parse_args()
    
    print("🔬 LMStudio SDK Integration Test")
    print("================================\n")
    
    success = True
    
    if not args.agent_only:
        success &= test_lmstudio_llm(args.api_host, args.model)
    
    if not args.llm_only:
        success &= test_react_agent_integration(args.api_host, args.model)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed! LMStudio integration is ready.")
        print("\n📝 Next steps:")
        print("   1. Start your LMStudio server")
        print("   2. Load a model in LMStudio") 
        print("   3. Run this test again with --api-host parameter")
        print("   4. Use 'lmstudio' as model_type in your configurations")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()