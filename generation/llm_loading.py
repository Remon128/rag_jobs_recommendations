from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import BitsAndBytesConfig
from transformers import pipeline


class LLM:
    def __init__(self):
        # Load pretrained Model and Tokenizer
        # model_id = "yam-peleg/Experiment24-7B"
        #model_id = "crumb/nano-mistral"
        #model_id = "mistralai/Mistral-7B-v0.1"
        model_id = "TinyLlama/TinyLlama_v1.1"

        bnb_config = BitsAndBytesConfig(
            activation_bits=4,  # Adjust as needed (e.g., 4 for lower precision, 16 for higher)
            weight_bits=4,  # Adjust as needed (e.g., 4 for lower precision, 16 for higher)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, config=bnb_config, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_id, config=bnb_config, device_map="auto")
        self.tokenizer = tokenizer
        self.model = model

    def instantiate_LLM_Reader(self):
        """This function creates LLM reader with model to ready for querying

        Returns:
            LLM reader object
        """
        LLM_Reader = pipeline(model=self.model,
                              tokenizer=self.tokenizer,
                              task="text-generation",
                              do_sample=True,
                              temperature=0.7,
                              return_full_text=True,
                              max_new_tokens=2000,
                              )

        return LLM_Reader


if __name__ == "__main__":
    llm = LLM()
    llm.instantiate_LLM_Reader()
