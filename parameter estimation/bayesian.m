function pred = bayesian(dataset,prior,mean)
dataset_size = size(dataset);
prior_size = size(prior);
data_num = dataset_size(1,1);
class_num = prior_size(1,1);
pred = ones(data_num,1);
for i = 1:data_num
    % calculate the posterior probability
    post_prob = zeros(class_num,1);
    %note that the variance across different classes is the same
    for class = 1:1:class_num
        post_prob(class) = mean(class)*dataset(i)-(0.5)*mean(class)*mean(class)'+log(prior(class));
    end
    [val,index]= max(post_prob);
    pred(i) = index;
end