import json
import os
from typing import List

import networkx as nx
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from annotated_text import annotated_text, parameters
from streamlit_extras import add_vertical_space as avs
from streamlit_extras.badges import badge

from scripts.similarity.get_score import *
from scripts.utils import get_filenames_from_dir
from scripts.utils.logger import init_logging_config
from scripts.similarity.get_score import get_score
from scripts.parsers import ParseJobDesc, ParseResume
from scripts.ReadPdf import read_single_pdf

init_logging_config()
cwd = find_path("Resume-Matcher")
config_path = os.path.join(cwd, "scripts", "similarity")

# Make sure to include this only once at the top of your code
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

parameters.SHOW_LABEL_SEPARATOR = False
parameters.BORDER_RADIUS = 3
parameters.PADDING = "0.5 0.25rem"

# Function to load the data provided by user


# Function create star graph
def create_star_graph(nodes_and_weights, title):
    # Create an empty graph
    G = nx.Graph()

    # Add the central node
    central_node = "resume"
    G.add_node(central_node)

    # Add nodes and edges with weights to the graph
    for node, weight in nodes_and_weights:
        G.add_node(node)
        G.add_edge(central_node, node, weight=weight * 100)

    # Get position layout for nodes
    pos = nx.spring_layout(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Rainbow",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    # Color node points by number of connections
    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        adjacencies = list(G.adj[node])  # changes here
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"{node}<br># of connections: {len(adjacencies)}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Show the figure
    st.plotly_chart(fig)

def create_annotated_text(
    input_string: str, word_list: List[str], annotation: str, color_code: str):
    # Tokenize the input string
    tokens = nltk.word_tokenize(input_string)

    # Convert the list to a set for quick lookups
    word_set = set(word_list)

    # Initialize an empty list to hold the annotated text
    annotated_text = []

    for token in tokens:
        # Check if the token is in the set
        if token in word_set:
            # If it is, append a tuple with the token, annotation, and color code
            annotated_text.append((token, annotation, color_code))
        else:
            # If it's not, just append the token as a string
            annotated_text.append(token)

    return annotated_text

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def tokenize_string(input_string):
    tokens = nltk.word_tokenize(input_string)
    return tokens

st.title(":blue[Resume Matcher]")
# Sidebar for user input
st.sidebar.title("Upload your files")
uploaded_resume = st.sidebar.file_uploader("Upload a Resume", type=['pdf', 'docx', 'txt'])
uploaded_job_desc = st.sidebar.file_uploader("Upload a Job Description", type=['pdf', 'docx', 'txt'])

# Saving the uploaded files
if uploaded_resume:
    with open("data/Resumes/" + uploaded_resume.name, "wb") as f:
        f.write(uploaded_resume.getbuffer())
    output = uploaded_resume.name
    st.success("Resume uploaded successfully.")
if uploaded_job_desc:
    with open("data/JobDescription/" + uploaded_job_desc.name, "wb") as f:
        f.write(uploaded_job_desc.getbuffer())
    output1 = uploaded_job_desc.name
    st.success("Job Description uploaded successfully.")

st.divider()
avs.add_vertical_space(1)

# converting the uploaded files to strings
resume_string = read_single_pdf(uploaded_resume.name)
st.write(resume_string)
job_desc_string = read_single_pdf(uploaded_job_desc.name)
st.write(job_desc_string)

resumeParser = ParseResume(resume_string)
output = resumeParser.get_JSON()
st.write(output)
jobParser = ParseJobDesc(job_desc_string)
output1 = jobParser.get_JSON()
st.write(output1)