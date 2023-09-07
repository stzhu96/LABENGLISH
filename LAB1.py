import cv2
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import base64

app = dash.Dash(__name__)
server =app.server

app.layout = html.Div([
    html.H1("LAB Feature Display"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drop or   ',
            html.A('Select Image')
        ]),
        style={
            'width': '50%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload')
])

def analyze_image(contents):
    # 将上传的内容解码为图像格式
    encoded_image = contents.split(',')[1]
    decoded_image = cv2.imdecode(np.frombuffer(
        bytes(base64.b64decode(encoded_image)), np.uint8), -1)

    # 将BGR图像转换为Lab图像
    lab_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2LAB)

    # 分别获取L、a、b三个通道的ndarray数据
    img_l = lab_image[:,:,0]
    img_a = lab_image[:,:,1]
    img_b = lab_image[:,:,2]

    # 按L、a、b三个通道分别计算颜色直方图
    l_hist = cv2.calcHist([lab_image],[0],None,[256],[0,256])
    a_hist = cv2.calcHist([lab_image],[1],None,[256],[0,256])
    b_hist = cv2.calcHist([lab_image],[2],None,[256],[0,256])
    m, dev = cv2.meanStdDev(lab_image)  # 计算L、a、b三通道的均值和方差

    # 计算三个通道的均值和标准差
    l_mean, l_std = np.mean(l_hist), np.std(l_hist)
    a_mean, a_std = np.mean(a_hist), np.std(a_hist)
    b_mean, b_std = np.mean(b_hist), np.std(b_hist)
    m, dev = cv2.meanStdDev(lab_image)

    return l_mean, l_std, a_mean, a_std, b_mean, b_std,m,dev

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output_image_upload(contents):
    if contents is not None:
        l_mean, l_std, a_mean, a_std, b_mean, b_std ,m,dev= analyze_image(contents)

        # 将图像显示在网页上
        return html.Div([
            html.H3('上传的图像：'),
            html.Img(src=contents, style={'width': '400px'}),
            #html.P(f'L通道均值：{l_mean:.2f}，标准差：{l_std:.2f}'),
            #html.P(f'a通道均值：{a_mean:.2f}，标准差：{a_std:.2f}'),
            #html.P(f'b通道均值：{b_mean:.2f}，标准差：{b_std:.2f}'),
            html.P(f'Lmean：{m.ravel().tolist()[2]:.2f}，L standard deviation：{dev.ravel().tolist()[2]:.2f}'),
            html.P(f'amean：{m.ravel().tolist()[1]:.2f}，a standard deviation：{dev.ravel().tolist()[1]:.2f}'),
            html.P(f'bmean：{m.ravel().tolist()[0]:.2f}，b standard deviation：{dev.ravel().tolist()[0]:.2f}'),
        ])


if __name__ == '__main__':
    app.run_server(debug=False)