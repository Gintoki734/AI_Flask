from flask import Flask, render_template, request, jsonify
import os
import markdown
import json
import pandas as pd
import subprocess  #Used to call local commands

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import time

app = Flask(__name__)

process = None

# File paths where the data will be stored
LOG_FILE = "user_page_time.csv"
LOG_QUIZ = "user_quiz_performance.csv"
LOG_LLM = "user_llm_time.csv"

# Define available courses - https://www.w3schools.com/python/python_dictionaries.asp
COURSES = {
    "Talking about yourself": {"content": "data/Talking about yourself.txt", "quiz": "data/Talking about yourself.json"},
    "Talking about family": {"content": "data/Talking about family.txt", "quiz": "data/Talking about family.json"},
    "Animals and pets": {"content": "data/Animals and pets.txt", "quiz": "data/Animals and pets.json"},
    "Classroom language": {"content": "data/Classroom language.txt", "quiz": "data/Classroom language.json"},
    "Weather and seasons": {"content": "data/Weather and seasons.txt", "quiz": "data/Weather and seasons.json"},
    "Numbers and numeracy": {"content": "data/Numbers and numeracy.txt", "quiz": "data/Numbers and numeracy.json"},
    "Days, months and dates": {"content": "data/Days, months and dates.txt", "quiz": "data/Days, months and dates.json"},
    "Colours": {"content": "data/Colours.txt", "quiz": "data/Colours.json"},
    "Clothes": {"content": "data/Clothes.txt", "quiz": "data/Clothes.json"},
    "Food and drink": {"content": "data/Food and drink.txt", "quiz": "data/Food and drink.json"},
    "Exploring the town": {"content": "data/Exploring the town.txt", "quiz": "data/Exploring the town.json"}
}

#Fuzzy Logic
# Variables to judge the whole quiz performance
quiz_score = ctrl.Antecedent(np.arange(0, 5, 1), 'quiz_score') 
total_time_on_quiz = ctrl.Antecedent(np.arange(0, 61, 1), 'total_time_on_quiz')  
total_hesitance = ctrl.Antecedent(np.arange(0, 12, 1), 'hesitant')  

# Define the output variable
result = ctrl.Consequent(np.arange(0, 101, 1), 'result')  # Range: 0 to 100

# Define membership functions for quiz_score
quiz_score['low'] = fuzz.trimf(quiz_score.universe, [0, 0, 2])  # More overlap at 2
quiz_score['medium'] = fuzz.trimf(quiz_score.universe, [1, 2, 3])  # More overlap around 2
quiz_score['high'] = fuzz.trimf(quiz_score.universe, [2, 4, 4])  # More overlap at 2-3

# Define membership functions for total_time_on_quiz
total_time_on_quiz['low'] = fuzz.trimf(total_time_on_quiz.universe, [0, 0, 15])
total_time_on_quiz['medium'] = fuzz.trimf(total_time_on_quiz.universe, [10, 20, 40])
total_time_on_quiz['high'] = fuzz.trimf(total_time_on_quiz.universe, [30, 45, 60])

# Membership functions for total_hesitation
total_hesitance['low'] = fuzz.trimf(total_hesitance.universe, [0, 0, 6])
total_hesitance['medium'] = fuzz.trimf(total_hesitance.universe, [5, 7, 8])
total_hesitance['high'] = fuzz.trimf(total_hesitance.universe, [7, 10, 11])


# Define membership functions for result
result['low'] = fuzz.trimf(result.universe, [0, 0, 50])
result['medium'] = fuzz.trimf(result.universe, [50, 65, 80])
result['high'] = fuzz.trimf(result.universe, [75, 90, 100])

#Low
rule1 = ctrl.Rule(quiz_score['low'] | total_hesitance['high'], result['low'])
rule7 = ctrl.Rule(quiz_score['low'] & total_time_on_quiz['low'], result['low'])
rule9 = ctrl.Rule(quiz_score['low'] & total_time_on_quiz['high'], result['low'])

#Medium
rule2 = ctrl.Rule(quiz_score['medium'] & total_hesitance['medium'], result['medium'])
rule5 = ctrl.Rule(quiz_score['medium'] & total_hesitance['low'], result['medium'])
rule6 = ctrl.Rule(quiz_score['medium'] & total_hesitance['high'], result['medium'])
rule8 = ctrl.Rule(quiz_score['high'] & total_time_on_quiz['high'], result['medium'])
rule10 = ctrl.Rule(quiz_score['medium'] & total_time_on_quiz['low'], result['medium'])
rule11 = ctrl.Rule(quiz_score['medium'] & total_time_on_quiz['high'], result['medium'])

#High
rule3 = ctrl.Rule(quiz_score['high'] & total_time_on_quiz['low'], result['high'])
rule4 = ctrl.Rule(quiz_score['high'] & total_hesitance['low'], result['high'])


# Create the fuzzy inference system
result_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
result_simulation = ctrl.ControlSystemSimulation(result_ctrl)


@app.route('/total_results', methods=['POST'])
def total_results():
    try:
        # Store the received values
        data = request.get_json()
        total_time = float(data.get('total_time', 0.0))
        score = int(data.get('total_score', 0))
        total_options_Clicked = int(data.get('optionsClicked', 0))
        total_question = data.get('totalQ', 0)

        # Input values
        result_simulation.input['quiz_score'] = score
        result_simulation.input['total_time_on_quiz'] = total_time
        result_simulation.input['hesitant'] = total_options_Clicked

        # Compute the result
        result_simulation.compute()
        # Call fuzzy logic function and get result
        fuzzy_result = round(result_simulation.output['result'], 2)

        # Decide on a prompt based on the result
        if fuzzy_result > 88:
            prompt = "I have performed excellent overall. Praise me in a few sentences"
        elif fuzzy_result < 33:
            prompt = f"I didn't perform well for these questions - {total_question}. Create a small course to help me understand"
        else:
            prompt = "I performed good overall. Motivate me to further get a better result in a few sentences"

        # Record time taked for the prompt to generate an output
        start_timer = time.time()
        output = prompt_ollama(prompt) # Ask the LLM
        end_timer = time.time()

        time_taken = round(end_timer - start_timer, 2) # Round the it to 2 decimal point

        prompt_word_count = len(prompt) # Count the number of letters used

        columns = ["SessionID", "Course", "Feedback", "Final_feedback", "Prompt_word_count", "Time_taken"]

        # Check if the file exists and read it, otherwise initialize an empty DataFrame
        if os.path.exists(LOG_LLM):
            df = pd.read_csv(LOG_LLM)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = {
            "SessionID": [len(df) + 1],
            "Course": [0],
            "Feedback": [0],
            "Final_feedback": [1],
            "Prompt_word_count": [prompt_word_count],
            "Time_taken": [time_taken]
        }

        # Function to write to the csv file
        write_to_csv(LOG_LLM, new_entry, columns)

        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

           


@app.route('/per_Q_results', methods=['POST'])
def per_Q_results():
    try:
        # Store the received values
        data = request.get_json()
        correct_answers = int(data.get('correct_answers', 0))
        options_Clicked = int(data.get('optionsClicked', 0))
        current_question = data.get('current_Q', "")
        options = data.get('C_option', [])
        answer = data.get('selected_answer', "")


        # If they get it right
        if correct_answers > 0:
            # They hesitated
            if options_Clicked > 2:
                prompt = f"I got this question correct '{current_question}' but I hesitated in these options - {options} the answer I selected was {answer}"
            else:
                prompt = f"I got this question correct for '{current_question}', the options were - {options} and the answer I selected was {answer}. Praise and motivate me in a few words"
        # If they get it wrong
        else:
            # They hesitated
            if options_Clicked > 2:
                prompt = f"I got this question incorrect '{current_question}' and I hesitated in these options - {options} the answer I selected was {answer}"
            else:
                prompt = f"I got this question incorrect '{current_question}' create a small explanation on why I got it wrong among these options - {options} the answer I selected was {answer}"

        # Record time taked for the prompt to generate an output
        start_timer = time.time()
        output = prompt_ollama(prompt) # Ask the LLM
        end_timer = time.time()

        time_taken = round(end_timer - start_timer,2) # Round the it to 2 decimal point

        prompt_word_count = len(prompt) # Count the number of letters used

        columns = ["SessionID", "Course", "Feedback", "Final_feedback", "Prompt_word_count", "Time_taken"]

        # Check if the file exists and read it, otherwise initialize an empty DataFrame
        if os.path.exists(LOG_LLM):
            df = pd.read_csv(LOG_LLM)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "SessionID": [len(df) + 1],
            "Course": [0],
            "Feedback": [1],
            "Final_feedback": [0],
            "Prompt_word_count": [prompt_word_count],
            "Time_taken": [time_taken]
        })

        # Function to write to the csv file
        write_to_csv(LOG_LLM, new_entry, columns)

        return jsonify({"response": output})

    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

           
#Function to interact with Ollama locally - https://github.com/ollama/ollama/issues/1474
def prompt_ollama(user_input):
    try:
        #Call Ollama using subprocess
        command = f"ollama run llama2"
        #Add the utf 8 reference here
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, encoding='utf-8')
        # Send the user input to Ollama
        process.stdin.write(user_input + "\n")
        process.stdin.flush()
        # Capture the full output after completion
        result, error = process.communicate()
        if process.returncode != 0:
            return f"Error: {error.strip()}"  # Capture the error
        else:
            return result.strip() #Capture the output from Ollama
    except Exception as e:
        return f"Exception: {str(e)}"
    

# https://stackoverflow.com/questions/47048906/convert-markdown-tables-to-html-tables-using-python
# Function to read and convert markdown content
def load_course_content(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return markdown.markdown(file.read(), extensions=["tables"])
    return "Content not available."

# Function to load quiz questions
def load_quiz(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"error": "Quiz not available."}

# Dispalys the first page
@app.route("/")
def home():
    return render_template("landingpage.html")

# Dispalys the home page
@app.route("/index")
def index():
    return render_template("index.html", courses=COURSES)

@app.route('/send_value', methods=['POST'])
def send_value():
    try:
        # Get the value sent from the user
        prompt = request.json.get('value')

        # Record time taked for the prompt to generate an output
        start_timer = time.time()
        output = prompt_ollama(prompt) # Ask the LLM
        end_timer = time.time()

        time_taken = round(end_timer - start_timer,2) # Round the it to 2 decimal point

        prompt_word_count = len(prompt) # Count the number of letters used

        columns = ["SessionID", "Course", "Feedback", "Final_feedback", "Prompt_word_count", "Time_taken"]

        # Check if the file exists and read it, otherwise initialize an empty DataFrame
        if os.path.exists(LOG_LLM):
            df = pd.read_csv(LOG_LLM)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "SessionID": [len(df) + 1],
            "Course": [1],
            "Feedback": [0],
            "Final_feedback": [0],
            "Prompt_word_count": [prompt_word_count],
            "Time_taken": [time_taken]
        })

        # Function to write to the csv file
        write_to_csv(LOG_LLM, new_entry, columns)
    
        return jsonify({'response': output})
    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

# Displays course content
@app.route("/course/<course_name>")
def course(course_name):
    if course_name in COURSES:
        content = load_course_content(COURSES[course_name]["content"])
        return render_template("course.html", course_name=course_name, content=content)
    return "Course not found", 404

# Displays quiz content for that specific course
@app.route("/quiz/<course_name>")
def quiz(course_name):
    if course_name in COURSES:
        quiz_data = load_quiz(COURSES[course_name]["quiz"])
        return render_template("quiz.html", course_name=course_name, quiz_data=quiz_data)
    return "Quiz not found", 404

# Writes the data collected from the course page into the csv file
@app.route('/log_time', methods=['POST'])
def log_time():
    # Handles any error occured
    try:
        # Getting the data in JSON format and retriving it and storing temporary data incase there is a void/missing data
        data = request.get_json()
        duration = data.get("duration", 0)
        course_name = data.get("courseName", "Unknown Course")
        date = data.get("date", "Unknown Date")
        time = data.get("time", "Unknown Time")

        columns = ["Session ID", "Time Spent (seconds)", "Course Name", "Date", "Time"]

        # Check if the file exists and read it otherwise initialize an empty DataFrame
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "Session ID": [len(df) + 1], 
            "Time Spent (seconds)": [duration],
            "Course Name": [course_name],
            "Date": [date],
            "Time": [time]
        })

        # Function to write to the csv file
        write_to_csv(LOG_FILE,new_entry,columns)

        return jsonify({"Message": "Quiz logged"}), 200

    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

@app.route('/log_quiz', methods=['POST'])
def log_quiz():
    # Handles any error occured
    try:
        # Getting the data in JSON format and retriving it and storing temporary data incase there is a void/missing data
        data = request.get_json()
        score = data.get("quiz_score", 0)
        course_name = data.get("courseName", "Unknown Course")
        date = data.get("date", "Unknown Date")
        time = data.get("time", "Unknown Time")
        optionClicked = data.get("clickCounts", [0] * 4)
        quiztime = data.get("quiztimetaken", [0] * 4)

        columns=[
                "Session ID", "Score", "Course Name", "Date", "Time",
                "OptionsClicked_Question_1", "OptionsClicked_Question_2", 
                "OptionsClicked_Question_3", "OptionsClicked_Question_4",
                "Time_Question_1", "Time_Question_2", "Time_Question_3", "Time_Question_4"
            ]

        # Check if the file exists and read it otherwise initialize an empty DataFrame
        if os.path.exists(LOG_QUIZ):
            df = pd.read_csv(LOG_QUIZ)  # Read the CSV file
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log            
        new_entry = pd.DataFrame({
            "Session ID": [len(df) + 1], 
            "Score": [score],
            "Course Name": [course_name],
            "Date" : [date],
            "Time": [time]
            })
        for i in range(4):
            new_entry[f"OptionsClicked_Question_{i+1}"] = optionClicked[i]
            new_entry[f"Time_Question_{i+1}"] = quiztime[i]

        # Function to write to the csv file
        write_to_csv(LOG_QUIZ, new_entry, columns)

        return jsonify({"Message": "Quiz logged"}), 200
    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500


def write_to_csv(file_path, new_entry, columns):
    # Ensure the file exists and read it, otherwise initialize a DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)

    # Append new entry and save it to CSV
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    #Call Ollama using subprocess
    command = f"ollama run llama2"
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, encoding='utf-8')
    app.run(debug=True)

