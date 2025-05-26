from Levenshtein import distance as levenshtein_distance
import ast
import json

class JsonOutputParser:
    def parseOutput(self, structuredResponse, outputDefinition): 
        try:
            structuredResponse = ast.literal_eval(structuredResponse)
        except:
            try:
                structuredResponse = json.loads(structuredResponse)
            except:
                print(f"Erro no formato: {structuredResponse}")
                return {}

        if outputDefinition is None:
            return structuredResponse

        isDict = False
        if isinstance(structuredResponse, dict):
            structuredResponse = [structuredResponse]
            isDict = True

        def get_most_similar_option(key, validOptionsList):
            min_distance = float('inf')  # Start with a large number for comparison
            best_option = None
            for option in validOptionsList:
                distance = levenshtein_distance(key, option)
                if distance <= 3 and distance < min_distance:
                    min_distance = distance
                    best_option = option
            return min_distance, best_option

        # Check and correct format to the valid names
        correctKeyNames = outputDefinition.keys()
        if isinstance(structuredResponse, list):
            for item in structuredResponse:
                if isinstance(item, dict):
                    keys_to_modify = []
                    # First collect the keys that need modification
                    for key in list(item.keys()):
                        if key not in correctKeyNames:
                            min_distance, best_option = get_most_similar_option(key, correctKeyNames)
                            if min_distance <= 3:
                                keys_to_modify.append((key, best_option))
                    
                    # Now modify the keys after iteration
                    for old_key, new_key in keys_to_modify:
                        item[new_key] = item.pop(old_key)

                    for correctKey in correctKeyNames:
                        # If a key is missing, add it with the default value
                        if correctKey not in item:
                            item[correctKey] = outputDefinition[correctKey][1]

                        #deal with the options type
                        validOptionsList = outputDefinition[correctKey][0]
                        if isinstance(validOptionsList, list) and len(validOptionsList) > 0:
                            if type(validOptionsList[0]) == str:
                                # Check if the key's value is in the valid options list, and correct it if needed
                                if item[correctKey] not in validOptionsList:
                                    #tries to correct it
                                    min_distance, best_option = get_most_similar_option(str(item[correctKey]), validOptionsList)
                                    if min_distance <= 3:
                                        item[correctKey] = best_option
                                    else:
                                        #set default
                                        item[correctKey] = outputDefinition[correctKey][1]
                        # if type is wrong, set to default
                        elif not isinstance(item[correctKey], outputDefinition[correctKey][0]):
                            item[correctKey] = outputDefinition[correctKey][1]
                        
                    #Remove keys that are not in the correctKeyNames
                    for key in list(item.keys()):
                        if key not in correctKeyNames:
                            item.pop(key)
        if isDict:
            structuredResponse = structuredResponse[0]
        return structuredResponse


if __name__ == "__main__":
    # Example usage
    response = str([{"CAMPO0": "asdasdasd",
                "CAMPO1": "SIM",
                "CAMPO2": "asdasdasd",
                "listaItems": ["asdasdasd"],
                "CAMPO4": "asdasdasd",
                "Selecao": "opção2",
                },
                {"CAMPO0": [],
                "CAMPO1": False,
                "CAMPO2": "asdasdasd",
                "listaItems": ["asdasdasd"],
                "CAMPO4": ["asdasdasd"],
                "Seleção": "opção 1",
                }])

    # {"Name": (type, default value)}
    outputDefinition = {"CAMPO0": (str,""),
                "CAMPO1": (bool,False),
                "CAMPO2": (str,""),
                "listaItems": (list,[]),
                "CAMPO4": (str,""),
                "Selecao": (["opção1", "opção2", "opção3"],"opção1"),
                }
    outputParser = JsonOutputParser()
    print(outputParser.parseOutput(response, outputDefinition))
