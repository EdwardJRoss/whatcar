import csv
from io import BytesIO
import uvicorn
import aiohttp
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageDataBunch, imagenet_stats
from fastai.vision.learner import create_cnn
from fastai.vision.models import resnet34
from fastai.vision.image import open_image
from urllib.request import urlretrieve
from pathlib import Path
import uuid

IMAGE_SIZE_LIMIT_BYTES = 8_500_000

IMAGE_DIR = Path('incoming/')
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_PATH = Path('labels.csv')

################################################################################
## Model Loading
################################################################################

def get_learner(classes):
    # TODO: Can we make this faster/lighter?
    data = ImageDataBunch.single_from_classes(".", classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data, resnet34, pretrained=False)
    learn.load('makemodel-392')
    return learn

def get_labels():
    with open('class_names.csv') as f:
        labels = [line for line in csv.DictReader(f)]
    return labels

LABELS = get_labels()
CLASSES = [c['class'] for c in LABELS]

LEARN = get_learner(CLASSES)


# TODO Class Labels

################################################################################
## Model scoring
################################################################################

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class, _pred_idx, outputs = LEARN.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(LEARN.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })

################################################################################
## Server
################################################################################

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()


def add_label(bytes, label, host, user_agent):
    name = str(uuid.uuid4())
    with open(LABEL_PATH, 'a') as f:
        print(f'{name}\t{label}\t{host}\t{user_agent}', file=f)
    with open(IMAGE_DIR / name, 'wb') as f:
        f.write(bytes)
    return JSONResponse({'status': 'ok'})

@app.route("/labels")
async def labels(request):
    return JSONResponse(LABELS)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    # Only read the first N bytes in case the file is too large and will hurt server
    bytes = await (data["file"].read(IMAGE_SIZE_LIMIT_BYTES))
    return predict_image_from_bytes(bytes)



@app.route("/submission", methods=["POST"])
async def submit(request):
    user_agent = request.headers['user-agent']
    host = request.client.host
    data = await request.form()
    # Only read the first N bytes in case the file is too large and will hurt server
    bytes = await (data["file"].read(IMAGE_SIZE_LIMIT_BYTES))
    label = data["class"]
    assert len(bytes) < IMAGE_SIZE_LIMIT_BYTES, "Image too large"
    assert label in CLASSES, f"Invalid class: {label}"
    # TODO: Some less hacky validation 
    assert ('\t' not in user_agent) and '\n' not in user_agent
    assert ('\t' not in host) and '\n' not in host
    return add_label(bytes, label, host, user_agent)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

@app.route('/css/{f}')
def css(request):
    f = request.path_params['f']
    return FileResponse('static/css/' + f)

@app.route('/img/{f}')
def image(request):
    f = request.path_params['f']
    return FileResponse('static/img/' + f)

@app.route("/")
def index(request):
    return FileResponse('index.html')

@app.route("/submit")
def index(request):
    return FileResponse('submit.html')

@app.route("/about")
def about(request):
    return FileResponse('about.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
