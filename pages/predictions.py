import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from joblib import load
from app import app
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            #### Enter a review : 
            """,style={'width': '90%', 'display': 'inline-block'}, className='mb-4'
        ),
        dcc.Textarea(id='tokens',placeholder='Example: the food was delicious and the atmosphere is nice',style={'height':100,'width': '90%', 'display': 'inline-block'},value='',className='mb-4'),
        dcc.Markdown(
            """
        
            #### My Movie Recommendations: 
            """,style={'width': '90%', 'display': 'inline-block'}, className='mb-4'
        ),
        html.Div(id='prediction-content', className='lead'),
        html.Div(id='prediction-content2', className='lead'),
        html.Div(id='prediction-content3', className='lead'),
        html.Div(id='prediction-content4', className='lead'),
        html.Div(id='prediction-content5', className='lead'),
        html.Img(src='assets/chrestaurant4.jpeg',style={'width': '90%', 'display': 'inline-block'}, className='img-fluid'),
        # dbc.FormText("Type something in the box above"),
               
        # for _ in ALLOWED_TYPES
    ],style={'display': 'inline-block'}
    # md=7,
)

column2 = dbc.Col(
    [   
    ]
)


layout = dbc.Row([column1])


@app.callback([
    Output('prediction-content', 'children'),
    ], 
    [Input('tokens','value')]
)


def predict(tokens):

    pipeline = pickle.load(open("./notebooks/pipe_01.pkl", "rb"))

    y_pred = pipeline.predict([tokens])

    return list(y_pred)