from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.1")

template = (
    "You are tasked with extracting specific information from the following text content {dom_content}"
    "Please follow these instructions carefully:\n\n"
    "1. **Extract information:** Only extract the information that directly matches the provided description {parse_description}."
    "2. **No extra content:** Don't include any additional text, comments or explanations in your response."
    "3. **Empty response:** If no information matches the description, return an empty string ('')."
    "4. **Direct data only:** Your output should contain only the data that's explicitly requested with no other text"
)

def parse_with_ollama(dom_chunks,parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results =[]

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
        print(f"Parsed batch {i} of {len(dom_chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)