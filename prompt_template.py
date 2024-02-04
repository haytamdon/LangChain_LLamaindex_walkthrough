from langchain_community.llms import Cohere, HuggingFaceHub
from langchain.prompts import PromptTemplate

def define_hf_model(repo_name, temperature):
    hf_llm = HuggingFaceHub(repo_id=repo_name, 
                            model_kwargs={'temperature':temperature})
    return hf_llm

def get_cohere_model():
    cohere_llm = Cohere()
    return cohere_llm

if __name__ == "__main__":
    hf_llm = define_hf_model("tiiuae/falcon-7b", temperature=0.1)
    cohere_llm = get_cohere_model()
    
    prompt_template = PromptTemplate(
        input_variables = ["length", "content"],
        template="Write a {length} line summary about the life of: {content}"
    )
    
    prompt = prompt_template.format(
        length="2",
        content="King Arthur"
    )
    
    print(prompt)
    
    falcon_response = hf_llm.invoke(prompt)
    cohere_response = cohere_llm.invoke(prompt)
    
    print(falcon_response)
    print(cohere_response)