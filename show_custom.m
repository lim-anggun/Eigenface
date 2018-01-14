"""
Created on Mon May 15 22:22:22 2017
@author: longang
"""
function show_custom(imList)
    faceW = 32; 
    faceH = 32; 
    
    num = size(imList,2); 
    for i=1:num 
        figure;
        imshow(reshape(imList(:,i),[faceH,faceW]), 'InitialMagnification', 300);
    end 
    colormap(gray);