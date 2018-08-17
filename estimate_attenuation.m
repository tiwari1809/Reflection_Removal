function [ck] = estimate_attenuation(I, dk_x, dk_y)

[size_x, size_y] = size(I);

%Harris Corner detector for interest points
corners = corner(I);
theta = 0.2;
filt_size = 18;
att = zeros(size(corners, 1));
weight = zeros(size(corners, 1));

for(i=1:size(corners, 1))
    origin_x = corners(i, 2);
    origin_y = corners(i, 1);
    
    if(origin_x > filt_size && origin_y > filt_size && origin_x <= size_x-filt_size && origin_y <= size_y-filt_size)
        patch1 = I(origin_x-filt_size:origin_x+filt_size, origin_y-filt_size:origin_y+filt_size);
    else 
        patch1 = [];
    end
    
    shifted_x = origin_x + dk_y;
    shifted_y = origin_y + dk_x;
    
    if(shifted_x > filt_size && shifted_y > filt_size && shifted_x <= size_x-filt_size && shifted_y <= size_y-filt_size)
        patch2 = I(shifted_x-filt_size:shifted_x+filt_size, shifted_y-filt_size:shifted_y+filt_size);
    else 
        patch2 = [];
    end
    
    if(isempty(patch1)||isempty(patch2))
        continue;
    end
    patch1=patch1-mean(patch1); patch2=patch2-mean(patch2);
    var1 = max(patch1(:)) - min(patch1(:));
    var2 = max(patch2(:)) - min(patch2(:));
    
    att(i) = (var1/var2);
    if(att(i) > 0 && att(i) < 1)
        weight(i) = exp((sum(sum(patch1.*patch2))/(sqrt(sum(sum(patch1.^2)))*sqrt(sum(sum(patch2.^2)))))/(2*theta^2));
    end
end
ck = sum(weight.*att)/sum(weight);

end

