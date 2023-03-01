%% a simple binary intertion function
% this function is to find the location (i.e., index) to insert some value
% (*val*) into a sorted vector (*arr*).

function y = bisect(arr, val)

ll = length(arr);

if val < arr(1)
    y = 1;
    return 
end

if val > arr(end)
    y = ll + 1;
    return
end

if ll == 1
    if val <= arr
        y = 1;
        return
    else
        y = 2;
        return
    end
end
       
mid = floor(ll / 2); 
if arr(mid) > val
    y = bisect(arr(1:mid-1), val);
else
    y = mid + bisect(arr(mid+1:end), val);
end