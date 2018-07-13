fid = fopen('test_name.csv', 'w');

for i =1:22700
    if test_ind(i)==1
        test_name = imgname(i);
        a = cell2mat(test_name);
        disp(a);
        fprintf(fid, '%s\n', a);
    end
end