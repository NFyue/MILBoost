import matplotlib.pyplot as plt



m = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 
42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 
84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 
122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 
158, 160, 162, 164, 166]
pred = [0.6274509803921569, 0.6372549019607843, 0.5686274509803921, 
0.5686274509803921, 0.5294117647058824, 0.46078431372549017, 
0.6470588235294118, 0.5490196078431373, 0.6078431372549019, 
0.6078431372549019, 0.5392156862745098, 0.5490196078431373, 
0.6078431372549019, 0.5588235294117647, 0.4215686274509804, 
0.5098039215686274, 0.5196078431372549, 0.4215686274509804, 
0.5882352941176471, 0.5980392156862745, 0.5784313725490197, 
0.6568627450980392, 0.6176470588235294, 0.5588235294117647, 
0.6078431372549019, 0.5098039215686274, 0.5784313725490197, 
0.47058823529411764, 0.5882352941176471, 0.5686274509803921, 
0.5294117647058824, 0.5098039215686274, 0.5784313725490197, 
0.5784313725490197, 0.6274509803921569, 0.5490196078431373, 
0.5, 0.6078431372549019, 0.6078431372549019, 0.5098039215686274, 
0.6764705882352942, 0.5392156862745098, 0.6568627450980392, 0.5392156862745098, 
0.5098039215686274, 0.4803921568627451, 0.5882352941176471, 0.6176470588235294, 
0.5098039215686274, 0.47058823529411764, 0.6176470588235294, 0.5392156862745098, 
0.6176470588235294, 0.5686274509803921, 0.4803921568627451, 0.46078431372549017, 
0.6078431372549019, 0.5588235294117647, 0.5196078431372549, 0.45098039215686275, 
0.6372549019607843, 0.6372549019607843, 0.5196078431372549, 0.6078431372549019, 0.5, 
0.5490196078431373, 0.6176470588235294, 0.5784313725490197, 0.5588235294117647, 
0.5490196078431373, 0.5196078431372549, 0.6078431372549019, 0.5882352941176471, 
0.5196078431372549, 0.6176470588235294, 0.6176470588235294, 0.5392156862745098, 
0.5882352941176471, 0.5686274509803921, 0.5392156862745098, 0.6274509803921569, 
0.5294117647058824, 0.5784313725490197, 0.5]
m2 = [0, 1, 2, 4, 8, 16, 32, 64, 128]
pred2 = [0.6274509803921569, 0.5980392156862745, 0.5588235294117647, 0.5588235294117647, 0.5392156862745098, 0.5980392156862745, 0.6274509803921569, 0.4215686274509804, 0.5686274509803921]
plt.plot(m2, pred2)
plt.ylabel('prediction accuracy')
plt.xlabel('m=numbers of changed features')
plt.savefig('prediction accuracy2.png')