from ollama import Client
from pydantic import BaseModel, create_model
from typing import Union, List, Dict, Any, Callable
import inspect
import json

class OllamaConversationLLMModel:
    def __init__(self, modelName: str = None, systemMessage: str = None, api_endpoint: str = 'http://localhost:11434'):
        super().__init__()
        self.modelName = modelName
        self.apiEndpoint = api_endpoint
        self.systemMessage = systemMessage
        self.history = []
        self.parametersSet = False

    def clear_history(self):
        """Reset conversation history while preserving the system message."""
        system_msg = None
        for msg in self.history:
            if msg['role'] == 'system':
                system_msg = msg
                break
        self.history = []
        if system_msg:
            self.history.append(system_msg)

    def setParameters(self):
        if self.systemMessage:
            self.history.insert(0, {'role': 'system', 'content': self.systemMessage})
            self.parametersSet = True

    def sendMessage(
        self,
        userMessage: str,
        expectsOutputParser: bool = False,
        outputDefinition: Dict = None,
        tools: List[Callable] = None
    ) -> Union[str, Dict, Any]:
        """
        Send a user message, manage conversation history, handle tool calls, 
        and return the assistant's response.
        """
        if not self.parametersSet:
            self.setParameters()
            
        if userMessage:
            self.history.append({'role': 'user', 'content': userMessage})

        messages = self.history.copy()
        final_response = None

        if tools:
            tool_schemas = [self._generate_tool_schema(func) for func in tools]
            system_msg = self._build_tool_system_message(tool_schemas)
            system_indices = [i for i, msg in enumerate(messages) if msg['role'] == 'system']
            if system_indices:
                messages[system_indices[0]]['content'] = system_msg
            else:
                messages.insert(0, {'role': 'system', 'content': system_msg})

        if expectsOutputParser and outputDefinition:
            DynamicModel = self._create_pydantic_model(outputDefinition)
            response = self.client.chat(
                model=self.modelName,
                messages=messages,
                format=DynamicModel.model_json_schema()
            )
            parsed = DynamicModel.model_validate_json(response.message['content'])
            final_response = parsed.dict()
            self.history.append({'role': 'assistant', 'content': json.dumps(final_response)})
        elif tools:
            FunctionCallModel = self._create_functioncall_model()
            response = self.client.chat(
                model=self.modelName,
                messages=messages,
                format=FunctionCallModel.model_json_schema()
            )
            func_call = FunctionCallModel.model_validate_json(response.message['content'])
            
            # Execute tool
            try:
                tool = next(t for t in tools if t.__name__ == func_call.function)
                result = tool(**func_call.arguments)
            except StopIteration:
                raise ValueError(f"Function {func_call.function} not found")
            except Exception as e:
                raise RuntimeError(f"Error executing {func_call.function}: {str(e)}")

            # Append tool interaction to history
            self.history.append({
                'role': 'assistant',
                'content': json.dumps({'function': func_call.function, 'arguments': func_call.arguments})
            })
            self.history.append({
                'role': 'tool',
                'content': json.dumps(result),
                'name': func_call.function
            })

            # Recursively continue conversation
            return self.sendMessage("", expectsOutputParser=expectsOutputParser, outputDefinition=outputDefinition, tools=tools)
        else:
            response = self.client.chat(
                model=self.modelName,
                messages=messages
            )
            final_response = response.message['content']
            self.history.append({'role': 'assistant', 'content': final_response})

        return final_response








