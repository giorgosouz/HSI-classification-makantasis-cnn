# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:58:18 2021

@author: vader
"""

import imageio as io
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from helper import collect_data,metrics,min_max,norm_img
from sklearn.preprocessing import MinMaxScaler
from tifffile import memmap



label_values = [
                            "Dense Urban Fabric",
                            "Mineral Extraction Sites",
                            "Non Irregated Arable Land",
                            "Fruit Trees",
                            "Olive Groves",
                            "Broad-leaved Forest",
                            "Coniferous Forest",
                            "Mixed Forest",
                            "Dense Sclerophyllous Vegetation",
                            "Sparce Sclerophyllous Vegetation",
                            "Sparcely Vegetated Areas",
                            "Rocks and Sand",
                            "Water",
                            "Coastal Water"]

img1=io.imread('TrainingSet/Dioni.tif')
img2=io.imread('TrainingSet/Loukia.tif')
gt1=io.imread('TrainingSet/Dioni_GT.tif')
gt2=io.imread('TrainingSet/Loukia_GT.tif')

Erato=io.imread('ValidationSet/Erato.tif')
Kirki=io.imread('ValidationSet/Kirki.tif')
Nefeli=io.imread('ValidationSet/Nefeli.tif')

min_val,max_val=min_max([img1,img2,Erato,Nefeli,Kirki])

img1=norm_img(img1, min_val, max_val)
X1,y1=collect_data(img1,gt1)

img2=norm_img(img2, min_val, max_val)
X2,y2=collect_data(img2,gt2)

X=np.concatenate((X1,X2))
y=np.concatenate((y1,y2))

N_BANDS=img1.shape[2]

del img1,img2,gt1,gt2


clf=LinearSVC(dual=False)
clf.fit(X,y)
y_pred=clf.predict(X)

report,cm=metrics(y_pred,y,label_values=label_values)
print(report,cm)
report.to_excel('svc_hy1_report.xlsx')
cm.to_excel('svc_hy1_report_cm.xlsx')


Erato=norm_img(Erato, min_val, max_val)
Kirki=norm_img(Kirki, min_val, max_val)
Nefeli=norm_img(Nefeli, min_val, max_val)

Erato_pred = clf.predict(((Erato.reshape(-1, N_BANDS)))).reshape(Erato.shape[:2])
Kirki_pred = clf.predict(((Kirki.reshape(-1, N_BANDS)))).reshape(Kirki.shape[:2])
Nefeli_pred = clf.predict(((Nefeli.reshape(-1, N_BANDS)))).reshape(Nefeli.shape[:2])

io.imwrite('Erato_pred_LinearSVC.tif',Erato_pred)
io.imwrite('Kirki_pred_LinearSVC.tif',Kirki_pred)
io.imwrite('Nefeli_pred_LinearSVC.tif',Nefeli_pred)



clf=RandomForestClassifier(n_jobs=-1)
clf.fit(X,y)
y_pred=clf.predict(X)
report,cm=metrics(y_pred,y,label_values=label_values)
print(report,cm)
report.to_excel('rf_hy1_report.xlsx')
cm.to_excel('rf_hy1_report_cm.xlsx')




Erato_pred = clf.predict(Erato.reshape(-1, N_BANDS)).reshape(Erato.shape[:2])
Kirki_pred = clf.predict((Kirki.reshape(-1, N_BANDS))).reshape(Kirki.shape[:2])
Nefeli_pred = clf.predict((Nefeli.reshape(-1, N_BANDS))).reshape(Nefeli.shape[:2])

io.imwrite('Erato_pred_RF.tif',Erato_pred)
io.imwrite('Kirki_pred_RF.tif',Kirki_pred)
io.imwrite('Nefeli_pred_RF.tif',Nefeli_pred)



label_values=[

'High Intensity Developped',
'Medium-Low Int.  Developped',
'Deciduous, Evergreen, Mixed Forest',
#    'Dead Trees (tree mortality)',
'Shrubland',
'Grassland-Pasture',
'Bareland',
#    'Wetlands',
'Water',
#    'Ice',
'Corn',
'Cotton',
'Cereals',
'Almonds',
'Grass Fodders',
'Vineyards-Grapes',
'Walnuts',
'Pistachios',
#    'Cherries',
#    'Tomato',
'Citrus',
#    'Rice',
'Fallow'
]





img1=memmap('Image1.tif', mode='r')
gt1=io.imread('GT1.tif')
img2=memmap('Image2.tif', mode='r')
gt2=io.imread('GT2.tif')
img5=memmap('Image5.tif', mode='r')
gt5=io.imread('GT5.tif')

N_BANDS=img1.shape[2]


img3=memmap('Image3.tif', mode='r')
img4=memmap('Image4.tif', mode='r')



min_val,max_val=min_max([img1,img2,img3,img4,img5])

img1=norm_img(img1, min_val, max_val)
X1,y1=collect_data(img1,gt1)

img2=norm_img(img2, min_val, max_val)
X2,y2=collect_data(img2,gt2)

img5=norm_img(img5, min_val, max_val)
X5,y5=collect_data(img5,gt5)




X=np.concatenate((X1,X2,X5))
y=np.concatenate((y1,y2,y5))

del X1,X2,X5,y1,y2,y5,img1,img2,img5






# idx = np.random.choice(np.arange(len(y)), 10000, replace=False)
# X=X[idx]
# y=y[idx]


clf_svc=LinearSVC(dual=False)
clf_svc.fit(X,y)
y_pred=clf_svc.predict(X)
report,cm=metrics(y_pred,y,label_values=label_values)
print(report,cm)
report.to_excel('svc_hy2_report.xlsx')
cm.to_excel('svc_hy2_report_cm.xlsx')


clf_rf=RandomForestClassifier(n_jobs=4)
clf_rf.fit(X,y)
y_pred=clf_rf.predict(X)
report,cm=metrics(y_pred,y,label_values=label_values)
print(report,cm)
report.to_excel('rf_hy2_report.xlsx')
cm.to_excel('rf_hy2_report_cm.xlsx')
del X,y



img3=memmap('Image3.tif', mode='r')
shape=img3.shape[:2]
img3=norm_img(img3, min_val, max_val)
img3=(img3.reshape(-1, N_BANDS))
img3_pred = clf_svc.predict(img3).reshape(shape)
img3_pred = np.asarray(img3_pred, dtype='int8')
io.imwrite('Image3_pred_SVC.tif',img3_pred)
img3_pred = clf_rf.predict(img3).reshape(shape)
img3_pred = np.asarray(img3_pred, dtype='int8')
io.imwrite('Image3_pred_RF.tif',img3_pred)
del img3


img4=memmap('Image4.tif', mode='r')
shape=img4.shape[:2]
img4=norm_img(img4, min_val, max_val)
img4=(img4.reshape(-1, N_BANDS))
img4_pred = clf_svc.predict(img4).reshape(shape)
img4_pred = np.asarray(img4_pred, dtype='int8')
io.imwrite('Image4_pred_SVC.tif',img4_pred)
img4_pred = clf_rf.predict(img4).reshape(shape)
img4_pred = np.asarray(img4_pred, dtype='int8')
io.imwrite('Image4_pred_RF.tif',img4_pred)
del img4



