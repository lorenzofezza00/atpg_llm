from models import get_model
import argparse

parser = argparse.ArgumentParser(description="Image Generator in loop")

# parser.add_argument("-txt_model", type=str, help="Text-to-text model", default=None)
# parser.add_argument("-eh", action="store_true", help="Enable history")
parser.add_argument("-autocomplete", action="store_true", help="Enable autocomplete", default=False)
parser.add_argument("-model", type=str, help="Model", default=None)
# parser.add_argument("-img_model", type=str, help="Text-to-image model", default=None)



# MODEL = 'sd_chat'

if __name__ == "__main__":

    args = parser.parse_args()
    MODEL = args.model
    if args.autocomplete:
        # --- AUTOCOMPLETE: usa i messaggi già definiti ---
        if MODEL == "gpt":
            gpt = get_model("gpt_oss")

            messages = [
                {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
            ]

            print(gpt.make_std_inference(messages))

        elif MODEL == "llama":
            lc = get_model("llama")

            messages = [
                {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
            ]

            print(lc.make_std_inference(messages))

        elif MODEL == "llama_chat":
            lc = get_model("llama_chat", system_prompt="You are a pirate chatbot who always responds in pirate speak!")

            res = lc.chat("Say hello to the world")
            print(f"Response: \n{res}")
            res = lc.chat("Be more pirate!")
            print(f"Response: \n{res}")
            res = lc.chat("Can you please tell me what I told you at the beginning?")
            print(f"Response: \n{res}")
            
        elif MODEL == "mistral":
            m = get_model("mistral", system_prompt="You are quantum computing expert")

            messages = [
                {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
            ]

            print(m.make_std_inference(messages))

        elif MODEL == "mistral_chat":
            m = get_model("mistral_chat", system_prompt="You are a pirate chatbot who always responds in pirate speak!")

            res = m.chat("Say hello to the world")
            print(f"Response: \n{res}")
            res = m.chat("Be more pirate!")
            print(f"Response: \n{res}")
            res = m.chat("Can you please tell me what I told you at the beginning?")
            print(f"Response: \n{res}")
        elif MODEL == "flux":
            m = get_model("flux", system_prompt="You are a digital artist specialized in scientific diagrams.")
            # singola generazione:
            single_path = m.generate("A clean, labeled diagram of a quantum circuit with 3 qubits.")
            print("Saved:", single_path)

            # generazione stepwise (es. storyboard o refining)
            prompts = [
                "Rough sketch: a quantum circuit with three qubits and two gates.",
                "Refine: add labels q0, q1, q2 and annotate Hadamard on first gate.",
                "Final: polished, high-resolution diagram suitable for publication."
            ]
            paths = m.generate_stepwise(prompts, num_inference_steps=25)
            for p in paths:
                print("Step saved:", p)

        elif MODEL == "flux_chat":
            m = get_model("flux_chat", system_prompt="You are a pirate chatbot who always responds in pirate speak!")
            p1 = m.chat("Create a pirate map showing an island, a mountain, and an X marking treasure.")
            print("Turn 1 image saved:", p1)
            p2 = m.chat("Now add a ship on the sea and a kraken on the horizon.")
            print("Turn 2 image saved:", p2)
            p3 = m.chat("Can you show me what I asked you to do at the beginning?")
            print("Turn 3 image saved:", p3)
            
        elif MODEL == "sd":
            m = get_model("sd", system_prompt="You are a digital artist specialized in scientific diagrams.")
            # singola generazione:
            single_path = m.generate("A clean, labeled diagram of a quantum circuit with 3 qubits.")
            print("Saved:", single_path)

            # generazione stepwise (es. storyboard o refining)
            prompts = [
                "Rough sketch: a quantum circuit with three qubits and two gates.",
                "Refine: add labels q0, q1, q2 and annotate Hadamard on first gate.",
                "Final: polished, high-resolution diagram suitable for publication."
            ]
            paths = m.generate_stepwise(prompts, num_inference_steps=25)
            for p in paths:
                print("Step saved:", p)

        elif MODEL == "sd_chat":
            
            m = get_model("sd_chat", system_prompt="You are an Automated Test Pattern Generator for circuits")
            p1 = m.chat("Create an instruction test library for testing multiplier units with floating point operations")
            print("Turn 1 image saved:", p1)
            p2 = m.chat("Now increase the fault critical rate")
            print("Turn 2 image saved:", p2)
            p3 = m.chat("The fault critcal rate obtained is 60%, increase it to 90%")
            print("Turn 3 image saved:", p3)
            
        elif MODEL == "sd_chat_llama":
            
            m = get_model("sd_chat_llama", system_prompt="You are an Automated Test Pattern Generator for circuits")
            p1 = m.chat("Create an instruction test library for testing multiplier units with floating point operations")
            print("Turn 1 image saved:", p1)
            p2 = m.chat("Now increase the fault critical rate")
            print("Turn 2 image saved:", p2)
            p3 = m.chat("The fault critcal rate obtained is 60%, increase it to 90%")
            print("Turn 3 image saved:", p3)

    else:
        # --- CHAT INTERATTIVA ---
        if MODEL == "gpt":
            gpt = get_model("gpt_oss")
            print("💬 Chat interattiva GPT (scrivi 'exit' per uscire)")
            while True:
                user_input = input("Tu: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                messages = [{"role": "user", "content": user_input}]
                response = gpt.make_std_inference(messages)
                print(f"🤖 GPT: {response}")

        elif MODEL == "llama":
            lc = get_model("llama")
            print("💬 Chat interattiva LLaMA (scrivi 'exit' per uscire)")
            while True:
                user_input = input("Tu: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                messages = [{"role": "user", "content": user_input}]
                response = lc.make_std_inference(messages)
                print(f"🦙 LLaMA: {response}")

        elif MODEL == "llama_chat":
            
            user_input = input("System prompt (e.g. \"You are a pirate chatbot who always responds in pirate speak!\"):").strip()
            lc = get_model("llama_chat", system_prompt=user_input)
            print("☠️ Chat interattiva LLaMA Chat (scrivi 'exit' per uscire)")
            while True:
                user_input = input("Tu: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                response = lc.chat(user_input)
                print(f"🏴‍☠️ LLaMA Chat: {response}")
