"""
Created on Mon May 15 22:22:22 2017
@author: longang
"""
function show_img(img)
    figure, imagesc(reshape(img,[32,32]));
    colormap(gray);