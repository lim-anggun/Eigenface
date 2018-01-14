"""
Created on Mon May 15 22:22:22 2017
@author: longang
"""
function show_some(imList, c, r, txt)
    faceW = 32; 
    faceH = 32; 
    numPerLine = c; %11,2
    ShowLine = r; 

    Y = zeros(faceH*ShowLine,faceW*numPerLine); 
    for i=0:ShowLine-1 
        for j=0:numPerLine-1 
            Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(imList(:,i*numPerLine+j+1),[faceH,faceW]); 
        end 
    end 
    figure; 
    imshow(Y, 'InitialMagnification', 300);
    title(txt);
    colormap(gray);