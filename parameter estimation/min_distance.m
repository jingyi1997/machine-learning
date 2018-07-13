function pred = min_distance(dataset,mean)
dataset_size = size(dataset);
mean_size = size(mean);
data_num = dataset_size(1,1);
class_num = mean_size(1,1); 
pred = ones(data_num,1);
for i = 1:data_num
    prob = ones(class_num,1);
    for j=1:class_num
        prob(j) = (1/2)*(dataset(i)-mean(j))*(dataset(i)-mean(j))'
    end
    [val,index]= min(prob);
    pred(i) = index;
end

    