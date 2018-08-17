function [C] = auto_corr_on_laplacian(I)
    H = fspecial('laplacian');
    lap = imfilter(I, H , 'replicate');
    OFFSET = 50; %+/- OFFSET on each side
    C = xcorr2(lap);
    [size_x , size_y] = size(C);
    C = C(floor((size_x+1)/2)-OFFSET:floor((size_x+1)/2)+OFFSET, floor((size_y+1)/2)-OFFSET:floor((size_y+1)/2)+OFFSET);
    
    %Visualization
    figure;
    subplot(1, 2, 1);
    imagesc(C);
    colorbar;
    subplot(1, 2, 2);
    mesh(C);
end

