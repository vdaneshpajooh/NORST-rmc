function  DisplayVideo( Data1, Data2, Data3, Data4, imSize, VideoName )


writerObj = VideoWriter(VideoName); % 
writerObj.FrameRate = 60; % How many frames per second.
open(writerObj); 
figure;

for i = 1: size(Data1,2)
    
   img1 = Data1(:,i);
   img1 = reshape(img1,imSize);
%    img1 = interp2(img1,sqrt(2));
   
   img2 = Data2(:,i);
   img2 = reshape(img2,imSize);
%    img2 = interp2(img2,sqrt(2));
   
   img3 = Data3(:,i);
   img3 = reshape(img3, imSize);
%    img3 = interp2(img3,sqrt(2));
   
   img4 = Data4(:,i);
   img4 = reshape(img4, imSize);
%    img4 = interp2(img4,sqrt(2));
    
    h = subplot('position',[0.01,0.50,0.47, 0.42]);
    imshow(img1/255);
    title(['data stream (time:',num2str(i),')'])
      
    h = subplot('position',[0.49,0.50,0.47, 0.42]);
    imshow(img2/255,[]);
    title('missing entries (\rho = 0.1)')
    
    h = subplot('position',[0.01,0.001,0.47, 0.42]);
    imshow(img3/255);
    title('data stream + missing entries')
    
    h = subplot('position',[0.49,0.001,0.47, 0.42]);
    imshow(img4/255);
    title('reconstructed data')
    
      
       frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
        writeVideo(writerObj, frame);
   
    
end

hold off
close(writerObj);
close all