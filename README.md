# Art'IA
Exploring art generation.

Data Source: mathildemagne.fr

Models explored:
1. Style Transfer (using VGG19)
2. AutoEncoder / Variational AutoEncoder


### Install the project

To install the project:

```bash
git clone git@github.com:casicoco/artia.git
cd artia
pip install -r requirements.txt
make clean install test 
```

## Testing style transfer 

Backend api is available in api.py
Frontend streamlit is available in app.py

You can run using:

```bash
make run_api
make run_streamlit
```

## AutoEncoder / Variational AutoEncoder
