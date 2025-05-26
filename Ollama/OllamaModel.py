from ollama import Client
from pydantic import BaseModel, create_model
from typing import Union, List, Dict, Any, Callable
import inspect
import json

class OllamaLLMModel:
    def __init__(self):
        self.modelName = None
        self.systemMessage = None
        self._api_endpoint = 'http://localhost:11434'
        self.client = Client(host=self._api_endpoint)
        
    @property
    def apiEndpoint(self):
        return self._api_endpoint
    
    @apiEndpoint.setter
    def apiEndpoint(self, value):
        self._api_endpoint = value
        self.client = Client(host=self._api_endpoint)

    def sendMessage(
        self, 
        userMessage: Union[str, List[str]], 
        expectsOutputParser: bool = False, 
        outputDefinition: Dict = None, 
        tools: List[Callable] = None,
        assistantFormat: bool = False
    ) -> Union[str, List[str], Dict, Any]:
        
        if assistantFormat:
            is_batch = False
        else:
            is_batch = isinstance(userMessage, list)
        
        messages_list = userMessage if is_batch else [userMessage]

        responses = []
        for msg in messages_list:
            messages = []
            if self.systemMessage:
                messages.append({'role': 'system', 'content': self.systemMessage})
            if assistantFormat:
                for roleContentDict in msg:
                    messages.append(roleContentDict)
            else:
                messages.append({'role': 'user', 'content': msg})

            if expectsOutputParser and outputDefinition:
                DynamicModel = self._create_pydantic_model(outputDefinition)
                response = self.client.chat(
                    model=self.modelName,
                    messages=messages,
                    format=DynamicModel.model_json_schema()
                )
                parsed = DynamicModel.model_validate_json(response.message.content)
                responses.append(parsed.dict())
                
            elif tools:
                tool_schemas = [self._generate_tool_schema(func) for func in tools]
                system_msg = self._build_tool_system_message(tool_schemas)
                messages = [{'role': 'system', 'content': system_msg}] + messages#[1:]
                
                FunctionCallModel = self._create_functioncall_model()
                response = self.client.chat(
                    model=self.modelName,
                    messages=messages,
                    format=FunctionCallModel.model_json_schema()
                )
                func_call = FunctionCallModel.model_validate_json(response.message.content)
                result = self._execute_tool(tools, func_call)
                responses.append(result)
                
            else:
                response = self.client.chat(
                    model=self.modelName,
                    messages=messages
                )
                responses.append(response.message['content'])
                
        return responses if is_batch else responses[0]

    def _create_pydantic_model(self, output_def):
        fields = {}
        for key, (type_, default) in output_def.items():
            fields[key] = (type_, Ellipsis if default is Ellipsis else default)
        return create_model('DynamicModel', **fields)

    def _generate_tool_schema(self, func):
        sig = inspect.signature(func)
        params = {
            'type': 'object',
            'properties': {},
            'required': []
        }
        for name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            params['properties'][name] = {'type': param_type.__name__}
            if param.default == inspect.Parameter.empty:
                params['required'].append(name)
        return {
            'name': func.__name__,
            'description': func.__doc__.strip() if func.__doc__ else "",
            'parameters': params
        }

    def _build_tool_system_message(self, tool_schemas):
        base_msg = self.systemMessage or "You are a helpful assistant."
        tools_desc = "\n".join(
            [f"{tool['name']}: {tool['description']}\nParameters: {json.dumps(tool['parameters'])}" 
             for tool in tool_schemas]
        )
        return f"{base_msg}\n\nAvailable tools:\n{tools_desc}\n\nRespond with JSON containing 'function' and 'arguments'."

    def _create_functioncall_model(self):
        class FunctionCall(BaseModel):
            function: str
            arguments: dict
        return FunctionCall

    def _execute_tool(self, tools, func_call):
        try:
            tool = next(t for t in tools if t.__name__ == func_call.function)
            return tool(**func_call.arguments)
        except StopIteration:
            raise ValueError(f"Function {func_call.function} not found")
        except Exception as e:
            raise RuntimeError(f"Error executing {func_call.function}: {str(e)}")
        

if __name__ == "__main__":
    pass
    # Usage example
    # ollama run deepseek-r1:7b
    # Normal query
    # osm = OllamaLLMModel()
    # osm.modelName = 'deepseek-r1:7b'
    # osm.systemMessage = "You are Jack Sparrow a helpful assistant."
    # print(osm.sendMessage("Hello what is your name?"))

    # # Batch processing
    # osm = OllamaLLMModel()
    # osm.modelName = 'deepseek-r1:7b'
    # osm.systemMessage = "Tell the main color of the fruit."
    # print(osm.sendMessage(["banana", "apple"]))

    # JSON parsing
    # osm2 = OllamaLLMModel()
    # osm2.modelName = "deepseek-r1:7b"
    # osm2.systemMessage = "You are a helpful assistant."
    # output_def = {
    #     "name": (str, ""),
    #     "age": (int, -1),
    #     "description": (str, "")
    # }
    # response = osm2.sendMessage(
    #     "Create a character sheet for Jack Sparrow",
    #     expectsOutputParser=True,
    #     outputDefinition=output_def
    # )
    # print(response)

    # Function calling
    # DOESN'T WORK VERY WELL DEPENDING ON THE MODEL
    # def multiply(a: float, b: float) -> float:
    #     """Multiply two numbers"""
    #     return a * b

    # def add(a: float, b: float) -> float:
    #     """Add two numbers"""
    #     return a + b

    # osm3 = OllamaLLMModel()
    # osm3.modelName = "deepseek-r1:7b"
    # response = osm3.sendMessage(
    #     "What is (754*432)+10? Please call the functions as exemplified in json format",
    #     tools=[multiply, add]
    # )
    # print(response)  # Should return 325738


