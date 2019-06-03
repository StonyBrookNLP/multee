import json
import re
import sys

# String used to indicate a blank
BLANK_STR = "___"


# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    applied = re.match(".*_+.*", fitb) is not None
    if not applied:
        # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + BLANK_STR
    return [fitb, applied]


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb, choice, highlight_answer=False) -> str:
    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    if highlight_answer:
        choice = " @@@answer {} answer@@@ ".format(choice)
    hypothesis = re.sub("__+", choice, fitb)
    return hypothesis


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str):
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?
        m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub("\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        return fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
    elif re.match(".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(" this[ \?]", " ___ ", question_str)

def question_answer_to_explicit_hypothesis(question_text, answer_text, implicit_hypothesis = None, highlight_answer=False):

    if not answer_text.strip() or answer_text == "\\":
        answer_text = "none" # In a case, answer is blank string!
    answer_text = answer_text.replace("\\", " ") # added later (during Race preprocessing)

    question_fitb, rule_applied = get_fitb_from_question(question_text)

    if not rule_applied and implicit_hypothesis:
        implicit_hypothesis_fitb = implicit_hypothesis + " " + BLANK_STR
        transformed_text = create_hypothesis(implicit_hypothesis_fitb, answer_text, highlight_answer)
    else:
        transformed_text = create_hypothesis(question_fitb, answer_text, highlight_answer)

    return transformed_text

if __name__ == "__main__":

    # question_text = ("George wants to warm his hands quickly by rubbing them. "
    #                  "Which skin surface will produce the most heat?")
    # answer_text = "dry palms"

    # question_text = ("In Us what happened and when?")
    # answer_text = "Event X and Event Y"
    # should be "In Us Event X happened and Event Y"

    question_text = ("Does he know the truth?")
    answer_text = "No"
    implicit_hypothesis = "he know the truth."

    hypothesis = question_answer_to_explicit_hypothesis(question_text, answer_text, implicit_hypothesis)

    print(hypothesis)
