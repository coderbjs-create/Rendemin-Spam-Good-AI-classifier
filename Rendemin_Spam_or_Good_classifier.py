#Read the license.txt file.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pathlib import Path
import sys
import csv

model = LinearSVC()
vectorizer = TfidfVectorizer(ngram_range=(1,2),lowercase=True,stop_words='english', min_df=2)
file = Path("C:\\Users\\User\\Desktop\\aiml\\datasets\\sms_spam_coll.csv")
feedback_file = Path("C:\\Users\\User\\Desktop\\aiml\\datasets\\sms_spam_coll_feedbacks.csv")

if not feedback_file.exists() or feedback_file.stat().st_size == 0:
    with open(feedback_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])

def unpack():
    dataset = pd.read_csv(file)

    labels = dataset["label"].tolist()
    msgs = dataset["message"].to_list()

    return labels, msgs

labels, msgs = unpack()

def load_feedback():
    try:
        df = pd.read_csv(feedback_file)
        return df["label"].tolist(), df["message"].tolist()
    except:
        return [], []
def train():
    fb_labels, fb_msgs = load_feedback()

    all_labels = labels + fb_labels
    all_msgs = msgs + fb_msgs

    msgs_vec = vectorizer.fit_transform(all_msgs)
    model.fit(msgs_vec, all_labels)
def user():
    train()
    print("\nHello, to the Rendemin Spam/Good SMS detector. Say 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Enter your text::>  ")
            another_inp = user_input

            if user_input.lower() == "quit":
                print("\nBye, come back again.\n")
                break
            else:
                user_vect = vectorizer.transform([user_input])
                answer = model.predict(user_vect)
                saved_answer = str(answer[0].strip().lower())

                print(f"\nDetected as: {answer[0].upper()}\n")

                feedback = input("Was our AI correct?:>  ").lower().replace(" ", "")
                print()

                if feedback == "yes":
                    with open(f"{feedback_file}", "a", encoding="utf-8", newline="") as fl:
                        writer = csv.writer(fl)
                        writer.writerow([saved_answer, another_inp])
                elif feedback == "no":
                    with open(f"{feedback_file}", "a", encoding="utf-8", newline="") as fl:
                        if saved_answer == "spam":
                            good_answer = "good"
                        else:
                            good_answer = "spam"

                        writer = csv.writer(fl)
                        writer.writerow([good_answer, another_inp])
                else:
                    print("\nInvalid Input, only can get 'yes' or 'no' for feedback.\n")
                    continue
                    
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye, come back again.\n")
            break

if __name__ == "__main__":
    try:
        user()
    except (KeyboardInterrupt, Exception):
        print("\nAn unexpected error happened..\n")
        sys.exit(0)