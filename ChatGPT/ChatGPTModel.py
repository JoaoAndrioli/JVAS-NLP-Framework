from openai import OpenAI
from Auxiliars.OutputParser import JsonOutputParser
from function_schema import get_function_schema
import json

class ChatGPTModel:
    def __init__(self, 
                 modelName = "", 
                 apiKey = "", 
                 tools=None,
                 systemMessage = "",
                 expectsOutputParser = False,
                 outputDefinition = None):
        self.modelName = modelName
        self.apiKey = apiKey
        if apiKey != "":
            self.client = OpenAI(api_key=apiKey)
        else:
            self.client = None
        self.tools = tools if tools else []
        self.systemMessage = systemMessage
        self.expectsOutputParser = expectsOutputParser
        self.outputDefinition = outputDefinition

    def setParameters(self):
        pass

    def sendMessage(self, userMessage, expectsOutputParser=None,
                 outputDefinition = None, tools = None, FunctionCallMessages = None):
        #if isinstance(userMessage, str):
        #    userMessage = [userMessage]
        
        if self.client is None:
            self.client = OpenAI(api_key=self.apiKey)

        # o1 model doesn't support system message, tool and json mode.
        iso1model = False
        if self.modelName in ["o1-preview", "o1-mini"]:
            iso1model = True

        if tools is None:
            tools = self.tools
        # Add tool functions if provided
        if tools and not iso1model:
            if not isinstance(tools, list):
                tools = [tools]
            schemas = []
            for tool in tools:
                schemas.append({"type": "function",
                                "function": get_function_schema(tool)})

        if type(userMessage) == str:
            isSingleMessage = True
            userMessage = [userMessage]
        else:
            isSingleMessage = False
        responses = []
        for currentUserMessage in userMessage:
            # Construct the message for the model
            if FunctionCallMessages is not None:
                messages = FunctionCallMessages
            else:
                if iso1model:
                    messages = [
                        {"role": "user", "content": currentUserMessage}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": self.systemMessage},
                        {"role": "user", "content": currentUserMessage}
                    ]

            # Build the API parameters
            completion_params = {
                "model": self.modelName,
                "messages": messages,
            }
            
            # Add tool functions if provided
            if tools and not iso1model:
                completion_params["tools"] = schemas
                #completion_params["tool_choice"] = "auto"

            if expectsOutputParser is None:
                expectsOutputParser = self.expectsOutputParser
            # Request JSON response format if expectsOutputParser is True
            if expectsOutputParser and not iso1model:
                completion_params["response_format"] = {"type": "json_object"}

            # Make the API call
            try:
                responseRaw = self.client.chat.completions.create(**completion_params)
            except Exception as e:
                print(f"Error: {e}")
                return str(e)

            if responseRaw.choices[0].finish_reason == "tool_calls":
                tool_call = responseRaw.choices[0].message.tool_calls[0]
                functionName = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                for func in tools:
                    if func.__name__ == functionName:
                        functionToUse = func
                        break
                print(f"Calling function: {functionName}.")
                responseFunction = functionToUse(**arguments)

                function_call_input_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {'id': tool_call.id,
                        'function': {
                            'arguments': tool_call.function.arguments,
                            'name': tool_call.function.name
                        },
                        'type': tool_call.type}
                        ]}

                function_call_result_message = {
                    "role": "tool",
                    "content": json.dumps(
                        arguments | {functionName+'_result': responseFunction}),
                    "tool_call_id": tool_call.id
                }

                completion_params['messages'].append(function_call_input_message)
                completion_params['messages'].append(function_call_result_message)

                if ((len(completion_params['messages'])-2)/2) > (len(tools)*3):
                #This means that all the functions were called 3 times each
                    print("Exiting function calling, found infinite loop on calling the function!!!")
                    return responseRaw
                
                responseRaw = self.sendMessage("", 
                                            expectsOutputParser = expectsOutputParser,
                                            outputDefinition = outputDefinition,
                                            tools = tools,
                                            FunctionCallMessages = completion_params['messages'])

            if FunctionCallMessages is not None:
                return responseRaw

            response = responseRaw.choices[0].message.content

            if expectsOutputParser:
                if outputDefinition is None:
                    outputDefinition = self.outputDefinition
                parser = JsonOutputParser()
                response = parser.parseOutput(response, outputDefinition)
            responses.append(response)

        if isSingleMessage:
            return responses[0]
        return responses




