from langchain_community.llms import Cohere, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.model_laboratory import ModelLaboratory

def define_hf_model(repo_name, temperature):
    hf_llm = HuggingFaceHub(repo_id=repo_name, 
                            model_kwargs={'temperature':temperature})
    return hf_llm

def get_cohere_model():
    cohere_llm = Cohere()
    return cohere_llm

if __name__ == "__main__":
    falcon_llm = define_hf_model("tiiuae/falcon-7b", temperature=0.1)
    mistral_llm = define_hf_model("mistralai/Mistral-7B-v0.1", temperature=0.1)
    cohere_llm = get_cohere_model()
    model_lab = ModelLaboratory.from_llms([falcon_llm,
                                            mistral_llm,
                                            cohere_llm])

    prompt_template = PromptTemplate(
        input_variables = ["length", "content"],
        template="Write a {length} line summary about the life of: {content}"
    )

    prompt = prompt_template.format(
        length="5",
        content="Nelson Mandela"
    )

    model_lab.compare(prompt)