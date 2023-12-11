import pandas as pd
import pickle
from flask import Flask, render_template, request, url_for
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from scipy import signal
import pandas as pd
import os
import glob
import shutil
import io
import base64


# Load the trained ML model from a pickle file
with open("stack_model.pkl", "rb") as f:
    model = pickle.load(f)

dic={}

def detect_mu_rhythm(EEG_signal, fs, low_freq, high_freq):

    # Apply a band-pass filter to extract the mu rhythm
    b, a = signal.butter(5, [low_freq / (fs / 2), high_freq / (fs / 2)], btype='band')
    mu_rhythm = signal.filtfilt(b, a, EEG_signal)

    f, Pxx = signal.periodogram(mu_rhythm, fs)
    # Threshold the mu power to determine if the mu rhythm is present
    mu_threshold = 90

    mu_power=max(Pxx)
    if mu_power > mu_threshold:
        return True
    else:
        return False


def spliting(x,y,h,j,EEG_signal,col):
    fs = 500
    l = []
    # Define the frequency range for the mu rhythm
    low_freq = 8
    high_freq = 30
    print(col)
    for i in range(x, y):

        split = EEG_signal.iloc[x:y]

        mu_present = detect_mu_rhythm(split, fs, low_freq, high_freq)
        if mu_present:
            # print(f"Mu rhythm is present in range {x} , {y}")
            # Names.append(f"{col}{f}{x}")
            l.append((x,y))

            # spliting(0,250,250, 1000,split)


        if y+h<j:

            x += h

            y += h

        if y+h>j:
            break
    print(l)
    dic[col]=l
df = pd.read_csv("test_data.csv")
# def plot_channel(eeg_data,channel):
#     plt.plot(eeg_data[channel])
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Voltage (uV)')
#     plt.title('EEG Signal: Channel {}'.format(channel))
#     return plt.gcf()

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/')
def about():
    # generate a URL to the 'index.html' file, with the '#my-section' anchor appended
    url = url_for('static', filename='index.html#about')
    return render_template('index.html', url=url)
@app.route('/Approaches')
def ap():
    return render_template('approach.html')
@app.route('/Mu Rhythm')
def mu():
    return render_template('mu rhythm.html')

@app.route('/Autism')
def autism():
    return render_template('autism.html')




@app.route("/classify", methods=["POST"])
def classify():
    # Get input values from the web form
    task = request.form["task"]
    subject = request.form["subject"]
    category = request.form["category"]
    channel = request.form["channel"]

    # print(task, subject, category, channel)
    # print(df.head())
    # Filter the DataFrame to find matching rows
    # index_value = df.index
    # print("index",index_value)
    sub = {"Subject 1": "TD1", "Subject 2": "TD2", "Subject 3": "AD2", }

    match_rows = df.loc[(df['Task'] == task) & (df['Subject'] == sub[subject]) & (df['Channel'] == channel) & ((df['Category']) == int(category))]
    #match_rows = df.loc[(df['Task'] == task)]
    # match_rows = match_rows.loc[match_rows['Subject'] == subject]
    # match_rows = match_rows.loc[match_rows['Category'] == category]
    # match_rows = match_rows.loc[match_rows['Channel'] == channel]

    # print(match_rows)

    sim = match_rows.drop(['File', 'Label', 'Task', 'Subject', 'Category', 'Channel'], axis=1)
    s=sim.values.tolist()
    ff=s[0]
    print("feat",ff[:5])
    features=ff[:5]
    print("row",match_rows.index[0])
    cols=["Alpha-Beta-Power (dB/Hz)",	"Alpha-Power (dB/Hz)",	"Beta-Power (dB/Hz)",	"Higuchi Factoral Dimension","Hurst Exponent"]
    data=zip(cols,features)

    if len(match_rows) == 0:
        # No matching rows found
        prediction = "No matching rows found"
    else:
        # Use the ML model to classify the matching rows
        X = df.drop(['File', 'Label', 'Task', 'Subject', 'Category', 'Channel'], axis=1)
        # print(X.values.tolist())
        y_pred = model.predict(X)
        print(y_pred)
        pred = y_pred[match_rows.index]
        l=["Autistic Child","Typically Developed"]
        print(pred)
        p=l[pred[0]]
        print(p)
        confidence_score = model.predict_proba(X)
        conf_score = np.max(confidence_score, axis=1)
        final_conf = conf_score[match_rows.index]
        final_conf = np.round(final_conf, 3)
    # print(features)
    if subject=="Subject 3":

        name = task + " " + sub[subject]
    else:
        name = task + " " + sub[subject] + " " + category
    eeg = pd.read_csv("full data/" + str(name) + ".csv")
    # fig = plot_channel(eeg_data,channel)

    # Save the plot to a PNG file
    # fig.savefig('static/plot.png')
    # plt.plot(eeg[channel])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Voltage (uV)')
    # plt.title('EEG Signal: Channel {}'.format(channel))
    #
    # img = io.BytesIO()
    # plt.savefig(img, format='png')
    # plt.close()
    # img.seek(0)
    #
    # # Encode the image file in base64 format
    # g = base64.b64encode(img.getvalue()).decode('utf-8')
    #
    # print(g)
    return render_template("result.html", prediction=p,data=data, confidence=final_conf[0])

@app.route('/up', methods=['POST'])
def upload_csv():
    if request.method == 'POST':
        graph_data = []
        file = request.files['file']
        d = pd.read_csv(file)
        for col in d.columns:
            EEG_signal = d[col]
            fs = 500

            # Define the frequency range for the mu rhythm
            low_freq = 8
            high_freq = 30

            # Detect the presence of the mu rhythm
            mu_present = detect_mu_rhythm(EEG_signal, fs, low_freq, high_freq)

            if mu_present:
                graph_data = []
                # print(f"Mu rhythm is present in {col}_{os.path.basename(f)}")

                spliting(0, 1000, 1000, 13001, EEG_signal, col)

                # print(dic)
                # If it exists, take the values from the column

                for key, value in dic.copy().items():
                    if len(value) == 0:
                        # If the list is empty, remove that key from the dictionary
                        del dic[key]
                print(dic)
                for i in dic:
                    print(i)
                    signal = d[i].values
                    colors = ['green']

                    # Plot the signal with colored intervals
                    fig, ax = plt.subplots()
                    ax.plot(signal)
                    ax.set_title(f"{i}")

                    for i, interval in enumerate(dic[i]):
                        ax.axvspan(interval[0], interval[1], color=colors[0], alpha=0.5)
                    # plt.show()
                    # Save the graph to a PNG image file
                    img = io.BytesIO()
                    plt.savefig(img, format='png')

                    plt.clf()
                    img.seek(0)

                    # Encode the image file in base64 format
                    graph_data.append(base64.b64encode(img.getvalue()).decode('utf-8'))

                    print(graph_data)
                # Render the template with the encoded image file
        return render_template('graph.html', graphs=graph_data)


if __name__ == '__main__':
    app.run(debug=True)
