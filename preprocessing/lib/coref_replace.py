import re
from collections import defaultdict

import spacy
from spacy.pipeline import SentenceSegmenter
import neuralcoref

# Old
# nlp = spacy.load('en_coref_lg')
# sentence_boundary = '@SentBoundary@'

# New
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

sentence_boundary = '@SentBoundary@'

raw_nlp = spacy.load('en')

def split_on_breaks(doc):
    start = 0
    seen_break = False
    for word in doc:
        if seen_break:
            yield doc[start:word.i-1]
            start = word.i
            seen_break = False
        elif word.text == sentence_boundary:
            seen_break = True
    if start < len(doc):
        yield doc[start:len(doc)]

sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_breaks)
nlp.add_pipe(sbd, first=True)

clean_text = lambda sent: sent.encode("ascii", errors="ignore").decode().replace('\\', '')


def inject_sentence_boundaries(paragraph, boundary_type="multirc"):
    if boundary_type == "multirc":
        paragraph = re.sub(r'(<br>)?(<b>Sent \d+: *<\/b>)', ' {} '.format(sentence_boundary), paragraph)
        paragraph = re.sub(r'<br> *$', '', paragraph)
        paragraph = re.sub(r'^ *{} *'.format(sentence_boundary), '', paragraph)
    elif boundary_type == "spacy":
        paragraph = " {} ".format(sentence_boundary).join([str(sent).strip() for sent in raw_nlp(paragraph).sents])
    return paragraph


def coref_replace(paragraph, boundary_type="multirc"):
    original_paragraph = inject_sentence_boundaries(clean_text(paragraph), boundary_type)
    processed_paragraph = nlp(original_paragraph)
    # make sure that doc has at least one coref clusters
    if not processed_paragraph._.has_coref:
        return [sent.text for sent in processed_paragraph.sents]
    # Get sentence boundaris first by pointer and cumsum
    pointer = 0
    sentence_markers = []
    for sentence in processed_paragraph.sents:
        sentence_len = len(sentence)
        sentence_markers.append((pointer, pointer+sentence_len-1))
        pointer += (sentence_len + 1)
    # Make empty sequences for each sentence:
    updated_token_sequences = [[] for index in range(len(sentence_markers))]
    sent_idx_to_replacements = defaultdict(list)
    for token in processed_paragraph:
        if token.text == sentence_boundary:
            continue
        token_sent_idx = [(marker[0] <= token.i <= marker[1])
                          for marker in sentence_markers].index(True)
        associated_nouns = []
        clusters = token._.coref_clusters
        for cluster in clusters:
            for mention in cluster.mentions:
                # A hack to change mentions: @SentBoundary@ Bhutto to Bhutto
                if mention[0].text == sentence_boundary:
                    mention = mention[1:]
                elif mention[-1].text == sentence_boundary:
                    mention = mention[:-1]
                if sentence_boundary in mention.text:
                    # IF hack didn't work, just continue.
                    continue
                mention_sent_idx = [(marker[0] <= mention[0].i <= marker[1])
                                    for marker in sentence_markers].index(True)
                # Don't make coref replace from the same sentence.
                if token_sent_idx == mention_sent_idx:
                    continue

                if token.pos_ == "PRON":
                    # same replacement shouldn't have been already been made in this sentence
                    already_replaced = sent_idx_to_replacements[token_sent_idx]
                    if any(replaced_mention.text == mention.text
                           for replaced_mention in already_replaced):
                        continue
                    if any(subtoken.pos_ == "NOUN" or subtoken.pos_ == "PROPN"
                           for subtoken in mention):
                        associated_nouns.append(mention)
                        sent_idx_to_replacements[token_sent_idx].append(mention)
        if associated_nouns:
            updated_token_sequences[token_sent_idx].append(associated_nouns[0])
        else:
            updated_token_sequences[token_sent_idx].append(token)
    updated_sentences = [" ".join([token.text for token in updated_token_sequence])
                         for updated_token_sequence in updated_token_sequences]
    return updated_sentences


def coref_replace_qa(question, answer):
    question_and_answer = " {} ".format(sentence_boundary).join([question, answer])
    coref_replaced_answer = coref_replace(question_and_answer)[1]
    return coref_replaced_answer


if __name__ == "__main__":

    # Try Paragraph Coref Replace
    paragraph = "<b>Sent 1: </b>(CNN) -- Air New Zealand's latest in-flight safety video, released Tuesday, is already another viral hit but is encountering some turbulence over its use of several bikini-clad Sports Illustrated models.<br><b>Sent 2: </b>View the video here Previous versions of the video -- starring anything from Hobbits to Bear Grylls to New Zealand's all conquering All Blacks rugby team -- have revolutionized the on-board safety message airlines deliver to passengers.<br><b>Sent 3: </b>The most recent effort though is being criticized by some as neither ground-breaking nor as creative, after the airline teamed up with Sports Illustrated magazine to produce what it's calling \"The world's most beautiful safety video.\"<br><b>Sent 4: </b>The \"Safety in Paradise\" video, which rolls out on Air New Zealand flights at the end of February, is beautifully shot and certainly cheerful and fun.<br><b>Sent 5: </b>It was filmed in the Cook Islands -- home to several stunning beaches -- and coincides with the 50th anniversary of Sports Illustrated's Swimsuit franchise.<br><b>Sent 6: </b>Earlier videos have been witty, clever and quirky but the paradise video combines a far less subtle use of eye-catching material -- using four of the planet's most beautiful, and scantily clad women, to deliver information to passengers.<br><b>Sent 7: </b>The models include Ariel Meredith, Chrissy Teigen, Hannah Davis and Jessica Gomes.<br><b>Sent 8: </b>Christie Brinkley makes a cameo.<br><b>Sent 9: </b>\"It seems that suddenly they are saying that my sexuality is all that matters about me,\" one critic, Massey University lecturer and feminist commentator Deborah Russell told the Sydney Morning Herald.<br><b>Sent 10: </b>Social media reaction to the video was predictably mixed, though the majority of commenters on Facebook and Twitter appeared to support the video -- and the women in it.<br><b>Sent 11: </b>Many praised Air New Zealand for using beautiful women to promote the Cook Islands and complimented the airline on its marketing prowess, given the mass of media attention now being given to the safety video.<br><b>Sent 12: </b>From the negative corner, while some commented they were appalled Air New Zealand would be so sexist, others said the Sports Illustrated version just isn't all that clever -- a disappointing follow up to the airline's creative safety videos of the past.<br>"
    print(coref_replace(paragraph))

    # Try QA Coref Replace
    # question = "Why did Mr. Thorndike feel a twinge of disappointment?"
    # answer = "Because the judge called him out"

    # answer_coref_replaced = coref_replace_qa(question, answer)
    # print(answer_coref_replaced)
    # # assert answer_coref_replaced == "Because the judge called Mr. Thorndike out"
