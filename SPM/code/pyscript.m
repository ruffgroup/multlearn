fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
ver,
try,
if ~(ismcc || isdeployed), addpath('~/spm12'); end;

if ~(ismcc || isdeployed), addpath('~/spm12'); end;
spm ver
,catch ME,
fprintf(2,'MATLAB code threw an exception:\n');
fprintf(2,'%s\n',ME.message);
if length(ME.stack) ~= 0, fprintf(2,'File:%s\nName:%s\nLine:%d\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;
end;