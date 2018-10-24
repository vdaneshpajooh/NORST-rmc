% Frames = 400 + [10,140,630,760];
Frames = [470,1059,1078,1154,1157];
% M2 = M + mu;
c = 0;
for iFrame = Frames
    x = reshape(L(:,iFrame),imSize);
%     c = c + 1;
%     subplot(10,10,c)
    h = figure;
    imshow(x/255);
    title(['frame #',num2str(iFrame)],'FontSize',5)
    hgexport(gcf,['BgFg_GRASTA_Frame',num2str(iFrame)],hgexport('factorystyle'),'Format','eps');
end
    