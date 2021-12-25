
clearvars;
%T = readtable("C:/Users/ypatia/Desktop/παραλληλα/Project2/hungary_chickenpox/hungary_chickenpox.csv",'NumHeaderLines',1);
T = readtable("C:/Users/ypatia/Desktop/παραλληλα/Project2/mnist_train.csv/mnist_train.csv");

% A=T(:,2:21);%chicken
% %B=T(:,6:69);
A = table2array(T);
A=A';
fileID = fopen('C:/Users/ypatia/Desktop/Docker-main/project2/file1.bin','w');
fwrite(fileID, size(A,1),'int64') ;
fwrite(fileID, size(A,2),'int64') ;
fwrite(fileID, A ,'double');
fclose(fileID);
size=[size(A,1),size(A,2)];
fileID = fopen('C:\Users\ypatia\Desktop\Docker-main\project2\file.bin','r');
B=fread(fileID,'double');
fclose(fileID);A