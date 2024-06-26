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
    [
    {{
        "diagnosis": "Bronchitis with appropriate severity level based on the conversation transcript",
        "procedure": "Chest X-ray",
        "australian_medicarecode": "49570"
    }},
    {{
        "diagnosis": "Ear infection with appropriate severity level based on the conversation transcript",
        "procedure": "Otoscopy",
        "australian_medicarecode": "16700"
    }}
    ]
    Please don't provide anything else apart from JSON output. Ensure the JSON output does not have keywords like ``` 
    or JSON or json.
    For each diagnosis and procedure, ensure there is only one medicare code.
    """

    # Request summary from Azure OpenAI
    response = client.chat.completions.create(
        model="gpt4o",
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=3500,
        messages=[
            {"role": "system", "content": "Assistant is a conversation summarizer between the doctor and patient."},
            {"role": "user", "content": prompt}
        ]
    )

    # Parse the response
    output = response.choices[0].message.content.strip()
    output = json.loads(output)
    # Load Medicare codes data
    with open("australianmedicarecodes.json", "r") as f:
        medicare_codes_data = json.load(f)

    # Create a dictionary from the MBS_XML data for quick lookup
    mbs_dict = {item["ItemNum"]: item for item in medicare_codes_data["MBS_XML"]["Data"]}
    filtered_list = []
    for record in output:
        medicare_code = record["australian_medicarecode"]
        if medicare_code in mbs_dict:
            additional_info = mbs_dict[medicare_code]
            record["ScheduleFee"] = additional_info.get("ScheduleFee", "N/A")
            record["Description"] = additional_info.get("Description", "N/A")
            filtered_list.append(record)

    return filtered_list


if __name__ == '__main__':
    with open("input_file.json", "r") as f:
        input_json = json.load(f)

    response = provide_australian_medicare_codes(input_json)
    print(response)
