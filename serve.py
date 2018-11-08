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

################################################################################
## Model Loading
################################################################################

def get_learner():
    with open('classes.csv') as f:
        classes = [line[0] for line in csv.reader(f)]

    # TODO: Can we make this faster/lighter?
    data = ImageDataBunch.single_from_classes(".", classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data, resnet34, pretrained=False)
    learn.load('car49-final1')
    return learn


LEARN = get_learner()

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




@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


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

@app.route("/about")
def about(request):
    return FileResponse('about.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
