import os
from pathlib import Path
from dotenv import load_dotenv
import markdown
from openai import OpenAI

# Import the TTS library and load the model
import torch
from TTS.api import TTS

# Initialize environment and the OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

########################
# 1) HELPER FUNCTIONS  #
########################

def load_file(path):
    """Simple utility to load any file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(prompt_template, topic, supplementary_info):
    """Use placeholders in 'prompt_template' for {topic} and {supplementary_info}."""
    return prompt_template.format(topic=topic, supplementary_info=supplementary_info)

def send_prompt_to_openai(prompt, model="gpt-4o"):
    """Sends the prompt to OpenAI and returns GPT's text response."""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def convert_markdown_to_html(markdown_text):
    """Converts GPT's Markdown output to HTML."""
    return markdown.markdown(markdown_text)

def inject_content_into_html(html_template, content_html):
    """Replaces {{CONTENT}} in the HTML template with the converted Markdown HTML."""
    return html_template.replace("{{CONTENT}}", content_html)

def extract_first_paragraph(markdown_text):
    """
    Splits the GPT response by blank lines to find paragraphs.
    Returns the first paragraph found. If none, returns empty string.
    """
    paragraphs = [p.strip() for p in markdown_text.split('\n\n') if p.strip()]
    if paragraphs:
        return paragraphs[0]
    return ""

##############################
# 2) TEXT-TO-SPEECH SETUP   #
##############################

# Load the XTTS model (change 'cuda' to 'cpu' if you have no GPU)
try:
    tts_model = TTS(model_name="tts_models/multilingual/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"Warning: Could not load XTTS model. {e}")
    tts_model = None

def synthesize_speech(text, speaker_wav, output_path, language="nl"):
    """
    Generates speech from the given text and reference speaker.
    Saves output as .wav to 'output_path'.
    """
    if not tts_model:
        print("TTS model not loaded; skipping audio synthesis.")
        return
    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav),
        language=language,
        file_path=str(output_path)
    )

##############################
# 3) MAIN SCRIPT FLOW       #
##############################

if __name__ == "__main__":

    # 🎙 Load XTTS-v2 Model
    tts = TTS(model_name="tts_models/multilingual/xtts_v2").to("cuda")  # Use GPU

    # 1. Adjust these variables as you need
    topic = "algorithmic bias"
    prompt_template_path = Path("../docs/prompt_template.txt")   # file with {topic} & {supplementary_info} placeholders
    supplementary_info_path = Path("docs/org_structure_amsterdam.txt")  # your supplementary info
    html_template_path = Path("../docs/template.html")            # your HTML skeleton with {{CONTENT}}
    speaker_wav_path = Path("../audio/Chriet_Titulaer.wav")        # reference speaker audio
    language_code = "nl"  # or 'en', 'de', etc.

    # 2. Output filenames in same folder, e.g. "output_algorithmic_bias.html" and "output_algorithmic_bias.wav"
    output_stem = f"output_{topic.replace(' ', '_')}"
    output_html_file = f"{output_stem}.html"
    output_wav_file = f"{output_stem}.wav"

    try:
        # 3. Load files from disk
        # prompt_template = load_file(prompt_template_path)
        # supplementary_info = load_file(supplementary_info_path)
        # html_template = load_file(html_template_path)

        # # 4. Build the GPT prompt
        # prompt = build_prompt(prompt_template, topic, supplementary_info)

        # # 5. Send prompt to OpenAI => get GPT's Markdown response
        # gpt_markdown_response = send_prompt_to_openai(prompt)

        # print("=== GPT Response (Markdown) ===\n")
        # print(gpt_markdown_response)

        # # 6. Convert Markdown -> HTML, then inject into template => final HTML
        # converted_html = convert_markdown_to_html(gpt_markdown_response)
        # final_html = inject_content_into_html(html_template, converted_html)

        # # 7. Write final HTML to disk
        # with open(output_html_file, "w", encoding="utf-8") as f:
        #     f.write(final_html)
        # print(f"\n✅ Response saved to HTML: {output_html_file}")

        # 8. Extract first paragraph => TTS
        # first_paragraph = extract_first_paragraph(gpt_markdown_response)
        first_paragraph = "Testing testing! Ik ben Chriet! Hoe is het eigenlijk met jullie allemaal?"
        if first_paragraph:
            print("\n=== First Paragraph ===\n", first_paragraph)
            # Generate speech
            synthesize_speech(
                text=first_paragraph,
                speaker_wav=speaker_wav_path,
                output_path=output_wav_file,
                language=language_code
            )
            print(f"✅ Synthesized speech saved to: {output_wav_file}")
        else:
            print("⚠ No text found for TTS (the GPT response was empty).")

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
