import threading

class LLMModelManager:
    def __init__(self, 
                 modelName = "", 
                 apiKey = "", 
                 tools=None,
                 systemMessage = "",
                 expectsOutputParser = False,
                 outputDefinition = None,
                 conversation_mode=False,
                 assistantFormat=True,
                 LLMType = "Ollama"):
        self.modelName = modelName
        self.apiKey = apiKey
        self.tools = tools if tools else []
        self.systemMessage = systemMessage
        self.expectsOutputParser = expectsOutputParser
        self.outputDefinition = outputDefinition
        self.conversation_mode = conversation_mode
        self.assistantFormat = assistantFormat

        self.LLMType = LLMType
 
        if self.LLMType == "Ollama":
            if self.conversation_mode:
                from Ollama.OllamaConversationModel import OllamaConversationLLMModel
                self.model = OllamaConversationLLMModel()
            else:
                from Ollama.OllamaModel import OllamaLLMModel
                self.model = OllamaLLMModel()
        elif self.LLMType == "ChatGPT":
            if self.conversation_mode:
                #from ChatGPT.OllamaConversationModel import OllamaConversationLLMModel
                #self.model = OllamaConversationLLMModel()
                pass
            else:
                from ChatGPT.ChatGPTModel import ChatGPTModel
                self.model = ChatGPTModel()
                self.model.apiKey = self.apiKey

        self.setParameters()

        # Dictionary to store responses from asynchronous calls, keyed by id.
        self.response = None
        self.processing = False
        self.lock = threading.Lock()

    def setParameters(self):
        #if self.LLMType == "Ollama":
        #    from Ollama.OllamaPullModel import pull_model
        #    pull_model(self.modelName)
        self.model.modelName = self.modelName
        self.model.systemMessage = self.systemMessage

    def _send_message_thread(self, message, expectsOutputParser, outputDefinition, tools, assistantFormat):
        """
        Worker function that calls the blocking sendMessage method and stores the result.
        """
        if self.conversation_mode:
            response = self.model.sendMessage(
                message,
                expectsOutputParser=expectsOutputParser,
                outputDefinition=outputDefinition,
                tools=tools
            )
        else:
            response = self.model.sendMessage(
                message,
                expectsOutputParser=expectsOutputParser,
                outputDefinition=outputDefinition,
                tools=tools,
                assistantFormat=assistantFormat
            )
        with self.lock:
            self.response = response

    def sendMessageAsync(self, message, expectsOutputParser=None, outputDefinition=None, tools=None, assistantFormat=None):
        """
        Initiates an asynchronous sendMessage call using a separate thread.
        The response is stored internally and can later be retrieved with getResponse(id).

        :param id: A unique identifier for this asynchronous call.
        :param message: The message to send.
        :param expectsOutputParser: (Optional) Override for expectsOutputParser.
        :param outputDefinition: (Optional) Override for outputDefinition.
        :param tools: (Optional) Override for tools.
        """
        
        if self.processing:
            return None

        if self.model.modelName == None or self.model.modelName == "":
            self.setParameters()
            
        expectsOutputParser = expectsOutputParser if expectsOutputParser is not None else self.expectsOutputParser
        outputDefinition = outputDefinition if outputDefinition is not None else self.outputDefinition
        tools = tools if tools is not None else self.tools
        assistantFormat = assistantFormat if assistantFormat is not None else self.assistantFormat

        # # Clear any previous response.
        # with self.lock:
        #     self.response = None

        thread = threading.Thread(
            target=self._send_message_thread, 
            args=(message, expectsOutputParser, outputDefinition, tools, assistantFormat)
        )
        self.processing = True
        thread.start()

    def getResponse(self):
        """
        Retrieves the response for the given id if available; otherwise, returns None.
        Once retrieved, the response is removed from storage.

        :param id: The unique identifier for the asynchronous call.
        :return: The response from the sendMessage call or None if not yet available.
        """
        with self.lock:
            currentResponse = self.response
            if self.response != None:
                self.processing = False
            self.response = None
        return currentResponse

    def sendMessage(self, 
                    userMessage, 
                    expectsOutputParser=None, 
                    outputDefinition = None, 
                    tools = None):
        if self.model.modelName == None or self.model.modelName == "":
            self.setParameters()
        
        if expectsOutputParser == None:
            expectsOutputParser = self.expectsOutputParser
        if outputDefinition == None:
            outputDefinition = self.outputDefinition
        if tools == None:
            tools = self.tools

        response = self.model.sendMessage(
            userMessage,
            expectsOutputParser=expectsOutputParser,
            outputDefinition=outputDefinition,
            tools=tools
        )

        return response

    def addAssistantMessage(self, message):
        self.model.history.append({'role': 'assistant', 'content': message})




