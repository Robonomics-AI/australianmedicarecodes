from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()  # Will load the env file

# Reading the API values from env file
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)


def provide_australian_medicare_codes(json_transcription):
    text = json_transcription["conversation"]

    prompt = f"""
    You are an AI assistant of a doctor. 
    Scenario: A doctor's appointment is recorded and transcribed. The doctor generates the summary notes in the
    in the form of Subjective , Objective, Assessment and Plan. You have been provided with the summary notes {text}.
    
    Goal: Extract the following information from the summary notes:

    Diagnosis: The medical condition identified by the doctor along with the severity.
    Procedure: Any medical procedure performed during the appointment.
    Australian Medicare Code: Generate the relevant Medicare Benefits Schedule (MBS) code associated 
    with the diagnosis and procedure. Please don't create medicare codes on your own.
    Desired Output:

    Remember the following pointers for each of the extracted information: 
    * ensure there is no diagnosis and procedure where the medicare code is not present.
    Here's an example:

    {{
        "diagnosis": "Bronchitis with appropriate severity level based on the conversation transcript",
        "procedure": "Chest X-ray",
        "australian_medicarecode": [49570]
    }},
    {{
        "diagnosis": "Ear infection with appropriate severity level based on the conversation transcript",
        "procedure": "Otoscopy",
        "australian_medicarecode": [16700]
    }}

    Please don't provide anything else apart from JSON output.
    """

    # Request summary from Azure OpenAI
    response = client.chat.completions.create(
        model="gmdevgpt4model",
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=20000,
        messages=[
            {"role": "system", "content": "Assistant is a conversation summarizer between the doctor and patient."},
            {"role": "user", "content": prompt}
        ]
    )

    # Parse the response
    output = response.choices[0].message.content.strip()
    summaries = []

    # Process each JSON object in the output
    for item in output.split("},"):
        item = item.strip() + "}" if not item.strip().endswith("}") else item.strip()
        try:
            summary_data = json.loads(item)
            summaries.append(summary_data)
        except json.JSONDecodeError:
            print("Invalid JSON format in the following item:", item)

    # Transform the summaries into the desired format
    transformed_data = []
    for item in summaries:
        diagnosis = item.get('diagnosis')
        procedure = item.get('procedure')
        medicare_codes = item.get('australian_medicarecode', [])

        for code in medicare_codes:
            transformed_data.append({
                'diagnosis': diagnosis,
                'procedure': procedure,
                'australian_medicarecode': code
            })

    # Load Medicare codes data
    with open("australianmedicarecodes.json", "r") as f:
        medicare_codes_data = json.load(f)

    # Create a dictionary from the MBS_XML data for quick lookup
    mbs_dict = {item["ItemNum"]: item for item in medicare_codes_data["MBS_XML"]["Data"]}

    # Filter and enrich the transformed data with Description and EMSNMaximumCap
    filtered_list_data = []
    for item in transformed_data:
        medicare_code = item["australian_medicarecode"]
        if medicare_code in mbs_dict:
            item.update({
                "Description": mbs_dict[medicare_code]["Description"],
                "ScheduleFee": mbs_dict[medicare_code]["ScheduleFee"]
            })
            filtered_list_data.append(item)

    return filtered_list_data


if __name__ == '__main__':
    with open("input_file.json", "r") as f:
        input_json = json.load(f)

    response = provide_australian_medicare_codes(input_json)
    print(response)

