ея$
±Ж
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ј
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
Щ
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8іЉ
Ц
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:A**
shared_nameconv2d_transpose_4/kernel
П
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:A*
dtype0
Ж
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
Т
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose/kernel
Л
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:*
dtype0
В
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:*
dtype0
Ц
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel
П
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
Ж
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0
Ц
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0**
shared_nameconv2d_transpose_2/kernel
П
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: 0*
dtype0
Ж
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
: *
dtype0
Ц
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`**
shared_nameconv2d_transpose_3/kernel
П
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:@`*
dtype0
Ж
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
И
total_confusion_matrixVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nametotal_confusion_matrix
Б
*total_confusion_matrix/Read/ReadVariableOpReadVariableOptotal_confusion_matrix*
_output_shapes

:*
dtype0
§
 Adam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*1
shared_name" Adam/conv2d_transpose_4/kernel/m
Э
4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/m*&
_output_shapes
:A*
dtype0
Ф
Adam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/m
Н
2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/m*
_output_shapes
:*
dtype0
М
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
Е
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_1/kernel/m
Й
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@ *
dtype0
А
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/m
Й
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
А
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m
Й
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
А
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
†
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose/kernel/m
Щ
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
:*
dtype0
Р
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/m
Й
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes
:*
dtype0
§
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/m
Э
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:*
dtype0
Ф
Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/m
Н
2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
§
 Adam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*1
shared_name" Adam/conv2d_transpose_2/kernel/m
Э
4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/m*&
_output_shapes
: 0*
dtype0
Ф
Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv2d_transpose_2/bias/m
Н
2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/m*
_output_shapes
: *
dtype0
§
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*1
shared_name" Adam/conv2d_transpose_3/kernel/m
Э
4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*&
_output_shapes
:@`*
dtype0
Ф
Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_3/bias/m
Н
2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/m*
_output_shapes
:@*
dtype0
§
 Adam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*1
shared_name" Adam/conv2d_transpose_4/kernel/v
Э
4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/v*&
_output_shapes
:A*
dtype0
Ф
Adam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/v
Н
2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/v*
_output_shapes
:*
dtype0
М
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
Е
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_1/kernel/v
Й
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@ *
dtype0
А
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/v
Й
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
А
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v
Й
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
А
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
†
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose/kernel/v
Щ
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
:*
dtype0
Р
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/v
Й
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes
:*
dtype0
§
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/v
Э
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:*
dtype0
Ф
Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/v
Н
2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
§
 Adam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*1
shared_name" Adam/conv2d_transpose_2/kernel/v
Э
4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/v*&
_output_shapes
: 0*
dtype0
Ф
Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv2d_transpose_2/bias/v
Н
2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/v*
_output_shapes
: *
dtype0
§
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*1
shared_name" Adam/conv2d_transpose_3/kernel/v
Э
4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*&
_output_shapes
:@`*
dtype0
Ф
Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_3/bias/v
Н
2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
ЌЛ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЗЛ
valueьКBшК BрК
л
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
Ж
layer_with_weights-0
layer-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
Ж
layer_with_weights-0
layer-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
Ж
layer_with_weights-0
layer-0
layer-1
 trainable_variables
!regularization_losses
"	variables
#	keras_api
Ж
$layer_with_weights-0
$layer-0
%layer-1
&trainable_variables
'regularization_losses
(	variables
)	keras_api
Ж
*layer_with_weights-0
*layer-0
+layer-1
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
Ж
4layer_with_weights-0
4layer-0
5layer-1
6trainable_variables
7regularization_losses
8	variables
9	keras_api
Ж
:layer_with_weights-0
:layer-0
;layer-1
<trainable_variables
=regularization_losses
>	variables
?	keras_api
Ж
@layer_with_weights-0
@layer-0
Alayer-1
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
®
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_rateFmљGmЊQmњRmјSmЅTm¬Um√VmƒWm≈Xm∆Ym«Zm»[m…\m ]mЋ^mћ_mЌ`mќFvѕGv–Qv—Rv“Sv”Tv‘Uv’Vv÷Wv„XvЎYvўZvЏ[vџ\v№]vЁ^vё_vя`vа
Ж
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15
F16
G17
 
Ж
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15
F16
G17
≠
alayer_regularization_losses
blayer_metrics
cnon_trainable_variables
trainable_variables

dlayers
regularization_losses
emetrics
	variables
 
h

Qkernel
Rbias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
R
jtrainable_variables
kregularization_losses
l	variables
m	keras_api

Q0
R1
 

Q0
R1
≠
nlayer_regularization_losses
olayer_metrics
pnon_trainable_variables
trainable_variables

qlayers
regularization_losses
rmetrics
	variables
h

Skernel
Tbias
strainable_variables
tregularization_losses
u	variables
v	keras_api
R
wtrainable_variables
xregularization_losses
y	variables
z	keras_api

S0
T1
 

S0
T1
≠
{layer_regularization_losses
|layer_metrics
}non_trainable_variables
trainable_variables

~layers
regularization_losses
metrics
	variables
l

Ukernel
Vbias
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
V
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api

U0
V1
 

U0
V1
≤
 Иlayer_regularization_losses
Йlayer_metrics
Кnon_trainable_variables
 trainable_variables
Лlayers
!regularization_losses
Мmetrics
"	variables
l

Wkernel
Xbias
Нtrainable_variables
Оregularization_losses
П	variables
Р	keras_api
V
Сtrainable_variables
Тregularization_losses
У	variables
Ф	keras_api

W0
X1
 

W0
X1
≤
 Хlayer_regularization_losses
Цlayer_metrics
Чnon_trainable_variables
&trainable_variables
Шlayers
'regularization_losses
Щmetrics
(	variables
l

Ykernel
Zbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
V
Юtrainable_variables
Яregularization_losses
†	variables
°	keras_api

Y0
Z1
 

Y0
Z1
≤
 Ґlayer_regularization_losses
£layer_metrics
§non_trainable_variables
,trainable_variables
•layers
-regularization_losses
¶metrics
.	variables
 
 
 
≤
 Іlayer_regularization_losses
®layer_metrics
©non_trainable_variables
0trainable_variables
™layers
1regularization_losses
Ђmetrics
2	variables
l

[kernel
\bias
ђtrainable_variables
≠regularization_losses
Ѓ	variables
ѓ	keras_api
V
∞trainable_variables
±regularization_losses
≤	variables
≥	keras_api

[0
\1
 

[0
\1
≤
 іlayer_regularization_losses
µlayer_metrics
ґnon_trainable_variables
6trainable_variables
Јlayers
7regularization_losses
Єmetrics
8	variables
l

]kernel
^bias
єtrainable_variables
Їregularization_losses
ї	variables
Љ	keras_api
V
љtrainable_variables
Њregularization_losses
њ	variables
ј	keras_api

]0
^1
 

]0
^1
≤
 Ѕlayer_regularization_losses
¬layer_metrics
√non_trainable_variables
<trainable_variables
ƒlayers
=regularization_losses
≈metrics
>	variables
l

_kernel
`bias
∆trainable_variables
«regularization_losses
»	variables
…	keras_api
V
 trainable_variables
Ћregularization_losses
ћ	variables
Ќ	keras_api

_0
`1
 

_0
`1
≤
 ќlayer_regularization_losses
ѕlayer_metrics
–non_trainable_variables
Btrainable_variables
—layers
Cregularization_losses
“metrics
D	variables
ec
VARIABLE_VALUEconv2d_transpose_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
≤
 ”layer_regularization_losses
‘layer_metrics
’non_trainable_variables
Htrainable_variables
÷layers
Iregularization_losses
„metrics
J	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEconv2d_transpose/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_transpose/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_2/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_2/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv2d_transpose_3/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_transpose_3/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
Ў0
ў1
Џ2
џ3

Q0
R1
 

Q0
R1
≤
 №layer_regularization_losses
Ёlayer_metrics
ёnon_trainable_variables
ftrainable_variables
яlayers
gregularization_losses
аmetrics
h	variables
 
 
 
≤
 бlayer_regularization_losses
вlayer_metrics
гnon_trainable_variables
jtrainable_variables
дlayers
kregularization_losses
еmetrics
l	variables
 
 
 

0
1
 

S0
T1
 

S0
T1
≤
 жlayer_regularization_losses
зlayer_metrics
иnon_trainable_variables
strainable_variables
йlayers
tregularization_losses
кmetrics
u	variables
 
 
 
≤
 лlayer_regularization_losses
мlayer_metrics
нnon_trainable_variables
wtrainable_variables
оlayers
xregularization_losses
пmetrics
y	variables
 
 
 

0
1
 

U0
V1
 

U0
V1
µ
 рlayer_regularization_losses
сlayer_metrics
тnon_trainable_variables
Аtrainable_variables
уlayers
Бregularization_losses
фmetrics
В	variables
 
 
 
µ
 хlayer_regularization_losses
цlayer_metrics
чnon_trainable_variables
Дtrainable_variables
шlayers
Еregularization_losses
щmetrics
Ж	variables
 
 
 

0
1
 

W0
X1
 

W0
X1
µ
 ъlayer_regularization_losses
ыlayer_metrics
ьnon_trainable_variables
Нtrainable_variables
эlayers
Оregularization_losses
юmetrics
П	variables
 
 
 
µ
 €layer_regularization_losses
Аlayer_metrics
Бnon_trainable_variables
Сtrainable_variables
Вlayers
Тregularization_losses
Гmetrics
У	variables
 
 
 

$0
%1
 

Y0
Z1
 

Y0
Z1
µ
 Дlayer_regularization_losses
Еlayer_metrics
Жnon_trainable_variables
Ъtrainable_variables
Зlayers
Ыregularization_losses
Иmetrics
Ь	variables
 
 
 
µ
 Йlayer_regularization_losses
Кlayer_metrics
Лnon_trainable_variables
Юtrainable_variables
Мlayers
Яregularization_losses
Нmetrics
†	variables
 
 
 

*0
+1
 
 
 
 
 
 

[0
\1
 

[0
\1
µ
 Оlayer_regularization_losses
Пlayer_metrics
Рnon_trainable_variables
ђtrainable_variables
Сlayers
≠regularization_losses
Тmetrics
Ѓ	variables
 
 
 
µ
 Уlayer_regularization_losses
Фlayer_metrics
Хnon_trainable_variables
∞trainable_variables
Цlayers
±regularization_losses
Чmetrics
≤	variables
 
 
 

40
51
 

]0
^1
 

]0
^1
µ
 Шlayer_regularization_losses
Щlayer_metrics
Ъnon_trainable_variables
єtrainable_variables
Ыlayers
Їregularization_losses
Ьmetrics
ї	variables
 
 
 
µ
 Эlayer_regularization_losses
Юlayer_metrics
Яnon_trainable_variables
љtrainable_variables
†layers
Њregularization_losses
°metrics
њ	variables
 
 
 

:0
;1
 

_0
`1
 

_0
`1
µ
 Ґlayer_regularization_losses
£layer_metrics
§non_trainable_variables
∆trainable_variables
•layers
«regularization_losses
¶metrics
»	variables
 
 
 
µ
 Іlayer_regularization_losses
®layer_metrics
©non_trainable_variables
 trainable_variables
™layers
Ћregularization_losses
Ђmetrics
ћ	variables
 
 
 

@0
A1
 
 
 
 
 
 
8

ђtotal

≠count
Ѓ	variables
ѓ	keras_api
I

∞total

±count
≤
_fn_kwargs
≥	variables
і	keras_api
I

µtotal

ґcount
Ј
_fn_kwargs
Є	variables
є	keras_api
L
Їtotal_confusion_matrix
Їtotal_cm
ї	variables
Љ	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ђ0
≠1

Ѓ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

∞0
±1

≥	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

µ0
ґ1

Є	variables
qo
VARIABLE_VALUEtotal_confusion_matrixEkeras_api/metrics/3/total_confusion_matrix/.ATTRIBUTES/VARIABLE_VALUE

Ї0

ї	variables
ЙЖ
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_transpose/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_transpose/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
О
serving_default_input_1Placeholder*1
_output_shapes
:€€€€€€€€€АА*
dtype0*&
shape:€€€€€€€€€АА
Џ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_9466
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Г
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*total_confusion_matrix/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpConst*O
TinH
F2D	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_10749
÷
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_transpose_4/kernelconv2d_transpose_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biastotalcounttotal_1count_1total_2count_2total_confusion_matrix Adam/conv2d_transpose_4/kernel/mAdam/conv2d_transpose_4/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/m Adam/conv2d_transpose_2/kernel/mAdam/conv2d_transpose_2/bias/m Adam/conv2d_transpose_3/kernel/mAdam/conv2d_transpose_3/bias/m Adam/conv2d_transpose_4/kernel/vAdam/conv2d_transpose_4/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/v Adam/conv2d_transpose_2/kernel/vAdam/conv2d_transpose_2/bias/v Adam/conv2d_transpose_3/kernel/vAdam/conv2d_transpose_3/bias/v*N
TinG
E2C*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_10957Ьч
Ќ”
ў
__inference__wrapped_model_8000
input_1:
6model_sequential_conv2d_conv2d_readvariableop_resource;
7model_sequential_conv2d_biasadd_readvariableop_resource>
:model_sequential_1_conv2d_1_conv2d_readvariableop_resource?
;model_sequential_1_conv2d_1_biasadd_readvariableop_resource>
:model_sequential_2_conv2d_2_conv2d_readvariableop_resource?
;model_sequential_2_conv2d_2_biasadd_readvariableop_resource>
:model_sequential_3_conv2d_3_conv2d_readvariableop_resource?
;model_sequential_3_conv2d_3_biasadd_readvariableop_resourceP
Lmodel_sequential_4_conv2d_transpose_conv2d_transpose_readvariableop_resourceG
Cmodel_sequential_4_conv2d_transpose_biasadd_readvariableop_resourceR
Nmodel_sequential_5_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceI
Emodel_sequential_5_conv2d_transpose_1_biasadd_readvariableop_resourceR
Nmodel_sequential_6_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceI
Emodel_sequential_6_conv2d_transpose_2_biasadd_readvariableop_resourceR
Nmodel_sequential_7_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceI
Emodel_sequential_7_conv2d_transpose_3_biasadd_readvariableop_resourceE
Amodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource<
8model_conv2d_transpose_4_biasadd_readvariableop_resource
identityИҐ/model/conv2d_transpose_4/BiasAdd/ReadVariableOpҐ8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpҐ.model/sequential/conv2d/BiasAdd/ReadVariableOpҐ-model/sequential/conv2d/Conv2D/ReadVariableOpҐ2model/sequential_1/conv2d_1/BiasAdd/ReadVariableOpҐ1model/sequential_1/conv2d_1/Conv2D/ReadVariableOpҐ2model/sequential_2/conv2d_2/BiasAdd/ReadVariableOpҐ1model/sequential_2/conv2d_2/Conv2D/ReadVariableOpҐ2model/sequential_3/conv2d_3/BiasAdd/ReadVariableOpҐ1model/sequential_3/conv2d_3/Conv2D/ReadVariableOpҐ:model/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpҐCmodel/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpҐ<model/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpҐEmodel/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ<model/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpҐEmodel/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ<model/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpҐEmodel/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpЁ
-model/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp6model_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02/
-model/sequential/conv2d/Conv2D/ReadVariableOpо
model/sequential/conv2d/Conv2DConv2Dinput_15model/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2 
model/sequential/conv2d/Conv2D‘
.model/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp7model_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.model/sequential/conv2d/BiasAdd/ReadVariableOpк
model/sequential/conv2d/BiasAddBiasAdd'model/sequential/conv2d/Conv2D:output:06model/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2!
model/sequential/conv2d/BiasAdd™
model/sequential/conv2d/ReluRelu(model/sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
model/sequential/conv2d/Reluф
&model/sequential/max_pooling2d/MaxPoolMaxPool*model/sequential/conv2d/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@@*
ksize
*
paddingVALID*
strides
2(
&model/sequential/max_pooling2d/MaxPoolй
1model/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:model_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype023
1model/sequential_1/conv2d_1/Conv2D/ReadVariableOp†
"model/sequential_1/conv2d_1/Conv2DConv2D/model/sequential/max_pooling2d/MaxPool:output:09model/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2$
"model/sequential_1/conv2d_1/Conv2Dа
2model/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;model_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2model/sequential_1/conv2d_1/BiasAdd/ReadVariableOpш
#model/sequential_1/conv2d_1/BiasAddBiasAdd+model/sequential_1/conv2d_1/Conv2D:output:0:model/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2%
#model/sequential_1/conv2d_1/BiasAddі
 model/sequential_1/conv2d_1/ReluRelu,model/sequential_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2"
 model/sequential_1/conv2d_1/ReluА
*model/sequential_1/max_pooling2d_1/MaxPoolMaxPool.model/sequential_1/conv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2,
*model/sequential_1/max_pooling2d_1/MaxPoolй
1model/sequential_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp:model_sequential_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1model/sequential_2/conv2d_2/Conv2D/ReadVariableOp§
"model/sequential_2/conv2d_2/Conv2DConv2D3model/sequential_1/max_pooling2d_1/MaxPool:output:09model/sequential_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2$
"model/sequential_2/conv2d_2/Conv2Dа
2model/sequential_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp;model_sequential_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2model/sequential_2/conv2d_2/BiasAdd/ReadVariableOpш
#model/sequential_2/conv2d_2/BiasAddBiasAdd+model/sequential_2/conv2d_2/Conv2D:output:0:model/sequential_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2%
#model/sequential_2/conv2d_2/BiasAddі
 model/sequential_2/conv2d_2/ReluRelu,model/sequential_2/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2"
 model/sequential_2/conv2d_2/ReluА
*model/sequential_2/max_pooling2d_2/MaxPoolMaxPool.model/sequential_2/conv2d_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2,
*model/sequential_2/max_pooling2d_2/MaxPoolй
1model/sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp:model_sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1model/sequential_3/conv2d_3/Conv2D/ReadVariableOp§
"model/sequential_3/conv2d_3/Conv2DConv2D3model/sequential_2/max_pooling2d_2/MaxPool:output:09model/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2$
"model/sequential_3/conv2d_3/Conv2Dа
2model/sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp;model_sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2model/sequential_3/conv2d_3/BiasAdd/ReadVariableOpш
#model/sequential_3/conv2d_3/BiasAddBiasAdd+model/sequential_3/conv2d_3/Conv2D:output:0:model/sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2%
#model/sequential_3/conv2d_3/BiasAddі
 model/sequential_3/conv2d_3/ReluRelu,model/sequential_3/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2"
 model/sequential_3/conv2d_3/ReluА
*model/sequential_3/max_pooling2d_3/MaxPoolMaxPool.model/sequential_3/conv2d_3/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2,
*model/sequential_3/max_pooling2d_3/MaxPoolє
)model/sequential_4/conv2d_transpose/ShapeShape3model/sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2+
)model/sequential_4/conv2d_transpose/ShapeЉ
7model/sequential_4/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model/sequential_4/conv2d_transpose/strided_slice/stackј
9model/sequential_4/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/sequential_4/conv2d_transpose/strided_slice/stack_1ј
9model/sequential_4/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model/sequential_4/conv2d_transpose/strided_slice/stack_2Ї
1model/sequential_4/conv2d_transpose/strided_sliceStridedSlice2model/sequential_4/conv2d_transpose/Shape:output:0@model/sequential_4/conv2d_transpose/strided_slice/stack:output:0Bmodel/sequential_4/conv2d_transpose/strided_slice/stack_1:output:0Bmodel/sequential_4/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model/sequential_4/conv2d_transpose/strided_sliceЬ
+model/sequential_4/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+model/sequential_4/conv2d_transpose/stack/1Ь
+model/sequential_4/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+model/sequential_4/conv2d_transpose/stack/2Ь
+model/sequential_4/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+model/sequential_4/conv2d_transpose/stack/3к
)model/sequential_4/conv2d_transpose/stackPack:model/sequential_4/conv2d_transpose/strided_slice:output:04model/sequential_4/conv2d_transpose/stack/1:output:04model/sequential_4/conv2d_transpose/stack/2:output:04model/sequential_4/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2+
)model/sequential_4/conv2d_transpose/stackј
9model/sequential_4/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model/sequential_4/conv2d_transpose/strided_slice_1/stackƒ
;model/sequential_4/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_4/conv2d_transpose/strided_slice_1/stack_1ƒ
;model/sequential_4/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_4/conv2d_transpose/strided_slice_1/stack_2ƒ
3model/sequential_4/conv2d_transpose/strided_slice_1StridedSlice2model/sequential_4/conv2d_transpose/stack:output:0Bmodel/sequential_4/conv2d_transpose/strided_slice_1/stack:output:0Dmodel/sequential_4/conv2d_transpose/strided_slice_1/stack_1:output:0Dmodel/sequential_4/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model/sequential_4/conv2d_transpose/strided_slice_1Я
Cmodel/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpLmodel_sequential_4_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02E
Cmodel/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpЫ
4model/sequential_4/conv2d_transpose/conv2d_transposeConv2DBackpropInput2model/sequential_4/conv2d_transpose/stack:output:0Kmodel/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:03model/sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
26
4model/sequential_4/conv2d_transpose/conv2d_transposeш
:model/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_4_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:model/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpҐ
+model/sequential_4/conv2d_transpose/BiasAddBiasAdd=model/sequential_4/conv2d_transpose/conv2d_transpose:output:0Bmodel/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2-
+model/sequential_4/conv2d_transpose/BiasAddћ
(model/sequential_4/conv2d_transpose/ReluRelu4model/sequential_4/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(model/sequential_4/conv2d_transpose/Reluґ
&model/sequential_4/up_sampling2d/ShapeShape6model/sequential_4/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2(
&model/sequential_4/up_sampling2d/Shapeґ
4model/sequential_4/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4model/sequential_4/up_sampling2d/strided_slice/stackЇ
6model/sequential_4/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6model/sequential_4/up_sampling2d/strided_slice/stack_1Ї
6model/sequential_4/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model/sequential_4/up_sampling2d/strided_slice/stack_2Ф
.model/sequential_4/up_sampling2d/strided_sliceStridedSlice/model/sequential_4/up_sampling2d/Shape:output:0=model/sequential_4/up_sampling2d/strided_slice/stack:output:0?model/sequential_4/up_sampling2d/strided_slice/stack_1:output:0?model/sequential_4/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:20
.model/sequential_4/up_sampling2d/strided_slice°
&model/sequential_4/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2(
&model/sequential_4/up_sampling2d/Constв
$model/sequential_4/up_sampling2d/mulMul7model/sequential_4/up_sampling2d/strided_slice:output:0/model/sequential_4/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2&
$model/sequential_4/up_sampling2d/mulЌ
=model/sequential_4/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor6model/sequential_4/conv2d_transpose/Relu:activations:0(model/sequential_4/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€*
half_pixel_centers(2?
=model/sequential_4/up_sampling2d/resize/ResizeNearestNeighborА
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis∞
model/concatenate/concatConcatV2Nmodel/sequential_4/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:03model/sequential_2/max_pooling2d_2/MaxPool:output:0&model/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€2
model/concatenate/concatЂ
+model/sequential_5/conv2d_transpose_1/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
:2-
+model/sequential_5/conv2d_transpose_1/Shapeј
9model/sequential_5/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model/sequential_5/conv2d_transpose_1/strided_slice/stackƒ
;model/sequential_5/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_5/conv2d_transpose_1/strided_slice/stack_1ƒ
;model/sequential_5/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_5/conv2d_transpose_1/strided_slice/stack_2∆
3model/sequential_5/conv2d_transpose_1/strided_sliceStridedSlice4model/sequential_5/conv2d_transpose_1/Shape:output:0Bmodel/sequential_5/conv2d_transpose_1/strided_slice/stack:output:0Dmodel/sequential_5/conv2d_transpose_1/strided_slice/stack_1:output:0Dmodel/sequential_5/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model/sequential_5/conv2d_transpose_1/strided_slice†
-model/sequential_5/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model/sequential_5/conv2d_transpose_1/stack/1†
-model/sequential_5/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-model/sequential_5/conv2d_transpose_1/stack/2†
-model/sequential_5/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model/sequential_5/conv2d_transpose_1/stack/3ц
+model/sequential_5/conv2d_transpose_1/stackPack<model/sequential_5/conv2d_transpose_1/strided_slice:output:06model/sequential_5/conv2d_transpose_1/stack/1:output:06model/sequential_5/conv2d_transpose_1/stack/2:output:06model/sequential_5/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model/sequential_5/conv2d_transpose_1/stackƒ
;model/sequential_5/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model/sequential_5/conv2d_transpose_1/strided_slice_1/stack»
=model/sequential_5/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model/sequential_5/conv2d_transpose_1/strided_slice_1/stack_1»
=model/sequential_5/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model/sequential_5/conv2d_transpose_1/strided_slice_1/stack_2–
5model/sequential_5/conv2d_transpose_1/strided_slice_1StridedSlice4model/sequential_5/conv2d_transpose_1/stack:output:0Dmodel/sequential_5/conv2d_transpose_1/strided_slice_1/stack:output:0Fmodel/sequential_5/conv2d_transpose_1/strided_slice_1/stack_1:output:0Fmodel/sequential_5/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model/sequential_5/conv2d_transpose_1/strided_slice_1•
Emodel/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_sequential_5_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02G
Emodel/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpС
6model/sequential_5/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput4model/sequential_5/conv2d_transpose_1/stack:output:0Mmodel/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0!model/concatenate/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
28
6model/sequential_5/conv2d_transpose_1/conv2d_transposeю
<model/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpEmodel_sequential_5_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<model/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp™
-model/sequential_5/conv2d_transpose_1/BiasAddBiasAdd?model/sequential_5/conv2d_transpose_1/conv2d_transpose:output:0Dmodel/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2/
-model/sequential_5/conv2d_transpose_1/BiasAdd“
*model/sequential_5/conv2d_transpose_1/ReluRelu6model/sequential_5/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*model/sequential_5/conv2d_transpose_1/ReluЉ
(model/sequential_5/up_sampling2d_1/ShapeShape8model/sequential_5/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2*
(model/sequential_5/up_sampling2d_1/ShapeЇ
6model/sequential_5/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model/sequential_5/up_sampling2d_1/strided_slice/stackЊ
8model/sequential_5/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/sequential_5/up_sampling2d_1/strided_slice/stack_1Њ
8model/sequential_5/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/sequential_5/up_sampling2d_1/strided_slice/stack_2†
0model/sequential_5/up_sampling2d_1/strided_sliceStridedSlice1model/sequential_5/up_sampling2d_1/Shape:output:0?model/sequential_5/up_sampling2d_1/strided_slice/stack:output:0Amodel/sequential_5/up_sampling2d_1/strided_slice/stack_1:output:0Amodel/sequential_5/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0model/sequential_5/up_sampling2d_1/strided_slice•
(model/sequential_5/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model/sequential_5/up_sampling2d_1/Constк
&model/sequential_5/up_sampling2d_1/mulMul9model/sequential_5/up_sampling2d_1/strided_slice:output:01model/sequential_5/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2(
&model/sequential_5/up_sampling2d_1/mul’
?model/sequential_5/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor8model/sequential_5/conv2d_transpose_1/Relu:activations:0*model/sequential_5/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€  *
half_pixel_centers(2A
?model/sequential_5/up_sampling2d_1/resize/ResizeNearestNeighborД
model/concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate/concat_1/axisЄ
model/concatenate/concat_1ConcatV2Pmodel/sequential_5/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:03model/sequential_1/max_pooling2d_1/MaxPool:output:0(model/concatenate/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€  02
model/concatenate/concat_1≠
+model/sequential_6/conv2d_transpose_2/ShapeShape#model/concatenate/concat_1:output:0*
T0*
_output_shapes
:2-
+model/sequential_6/conv2d_transpose_2/Shapeј
9model/sequential_6/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model/sequential_6/conv2d_transpose_2/strided_slice/stackƒ
;model/sequential_6/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_6/conv2d_transpose_2/strided_slice/stack_1ƒ
;model/sequential_6/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_6/conv2d_transpose_2/strided_slice/stack_2∆
3model/sequential_6/conv2d_transpose_2/strided_sliceStridedSlice4model/sequential_6/conv2d_transpose_2/Shape:output:0Bmodel/sequential_6/conv2d_transpose_2/strided_slice/stack:output:0Dmodel/sequential_6/conv2d_transpose_2/strided_slice/stack_1:output:0Dmodel/sequential_6/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model/sequential_6/conv2d_transpose_2/strided_slice†
-model/sequential_6/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2/
-model/sequential_6/conv2d_transpose_2/stack/1†
-model/sequential_6/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-model/sequential_6/conv2d_transpose_2/stack/2†
-model/sequential_6/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2/
-model/sequential_6/conv2d_transpose_2/stack/3ц
+model/sequential_6/conv2d_transpose_2/stackPack<model/sequential_6/conv2d_transpose_2/strided_slice:output:06model/sequential_6/conv2d_transpose_2/stack/1:output:06model/sequential_6/conv2d_transpose_2/stack/2:output:06model/sequential_6/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model/sequential_6/conv2d_transpose_2/stackƒ
;model/sequential_6/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model/sequential_6/conv2d_transpose_2/strided_slice_1/stack»
=model/sequential_6/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model/sequential_6/conv2d_transpose_2/strided_slice_1/stack_1»
=model/sequential_6/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model/sequential_6/conv2d_transpose_2/strided_slice_1/stack_2–
5model/sequential_6/conv2d_transpose_2/strided_slice_1StridedSlice4model/sequential_6/conv2d_transpose_2/stack:output:0Dmodel/sequential_6/conv2d_transpose_2/strided_slice_1/stack:output:0Fmodel/sequential_6/conv2d_transpose_2/strided_slice_1/stack_1:output:0Fmodel/sequential_6/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model/sequential_6/conv2d_transpose_2/strided_slice_1•
Emodel/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_sequential_6_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02G
Emodel/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpУ
6model/sequential_6/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput4model/sequential_6/conv2d_transpose_2/stack:output:0Mmodel/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0#model/concatenate/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
28
6model/sequential_6/conv2d_transpose_2/conv2d_transposeю
<model/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpEmodel_sequential_6_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<model/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp™
-model/sequential_6/conv2d_transpose_2/BiasAddBiasAdd?model/sequential_6/conv2d_transpose_2/conv2d_transpose:output:0Dmodel/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2/
-model/sequential_6/conv2d_transpose_2/BiasAdd“
*model/sequential_6/conv2d_transpose_2/ReluRelu6model/sequential_6/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2,
*model/sequential_6/conv2d_transpose_2/ReluЉ
(model/sequential_6/up_sampling2d_2/ShapeShape8model/sequential_6/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2*
(model/sequential_6/up_sampling2d_2/ShapeЇ
6model/sequential_6/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model/sequential_6/up_sampling2d_2/strided_slice/stackЊ
8model/sequential_6/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/sequential_6/up_sampling2d_2/strided_slice/stack_1Њ
8model/sequential_6/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/sequential_6/up_sampling2d_2/strided_slice/stack_2†
0model/sequential_6/up_sampling2d_2/strided_sliceStridedSlice1model/sequential_6/up_sampling2d_2/Shape:output:0?model/sequential_6/up_sampling2d_2/strided_slice/stack:output:0Amodel/sequential_6/up_sampling2d_2/strided_slice/stack_1:output:0Amodel/sequential_6/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0model/sequential_6/up_sampling2d_2/strided_slice•
(model/sequential_6/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model/sequential_6/up_sampling2d_2/Constк
&model/sequential_6/up_sampling2d_2/mulMul9model/sequential_6/up_sampling2d_2/strided_slice:output:01model/sequential_6/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2(
&model/sequential_6/up_sampling2d_2/mul’
?model/sequential_6/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor8model/sequential_6/conv2d_transpose_2/Relu:activations:0*model/sequential_6/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
half_pixel_centers(2A
?model/sequential_6/up_sampling2d_2/resize/ResizeNearestNeighborД
model/concatenate/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate/concat_2/axisі
model/concatenate/concat_2ConcatV2Pmodel/sequential_6/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0/model/sequential/max_pooling2d/MaxPool:output:0(model/concatenate/concat_2/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@`2
model/concatenate/concat_2≠
+model/sequential_7/conv2d_transpose_3/ShapeShape#model/concatenate/concat_2:output:0*
T0*
_output_shapes
:2-
+model/sequential_7/conv2d_transpose_3/Shapeј
9model/sequential_7/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model/sequential_7/conv2d_transpose_3/strided_slice/stackƒ
;model/sequential_7/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_7/conv2d_transpose_3/strided_slice/stack_1ƒ
;model/sequential_7/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model/sequential_7/conv2d_transpose_3/strided_slice/stack_2∆
3model/sequential_7/conv2d_transpose_3/strided_sliceStridedSlice4model/sequential_7/conv2d_transpose_3/Shape:output:0Bmodel/sequential_7/conv2d_transpose_3/strided_slice/stack:output:0Dmodel/sequential_7/conv2d_transpose_3/strided_slice/stack_1:output:0Dmodel/sequential_7/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model/sequential_7/conv2d_transpose_3/strided_slice†
-model/sequential_7/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2/
-model/sequential_7/conv2d_transpose_3/stack/1†
-model/sequential_7/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2/
-model/sequential_7/conv2d_transpose_3/stack/2†
-model/sequential_7/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2/
-model/sequential_7/conv2d_transpose_3/stack/3ц
+model/sequential_7/conv2d_transpose_3/stackPack<model/sequential_7/conv2d_transpose_3/strided_slice:output:06model/sequential_7/conv2d_transpose_3/stack/1:output:06model/sequential_7/conv2d_transpose_3/stack/2:output:06model/sequential_7/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model/sequential_7/conv2d_transpose_3/stackƒ
;model/sequential_7/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model/sequential_7/conv2d_transpose_3/strided_slice_1/stack»
=model/sequential_7/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model/sequential_7/conv2d_transpose_3/strided_slice_1/stack_1»
=model/sequential_7/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model/sequential_7/conv2d_transpose_3/strided_slice_1/stack_2–
5model/sequential_7/conv2d_transpose_3/strided_slice_1StridedSlice4model/sequential_7/conv2d_transpose_3/stack:output:0Dmodel/sequential_7/conv2d_transpose_3/strided_slice_1/stack:output:0Fmodel/sequential_7/conv2d_transpose_3/strided_slice_1/stack_1:output:0Fmodel/sequential_7/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model/sequential_7/conv2d_transpose_3/strided_slice_1•
Emodel/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_sequential_7_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype02G
Emodel/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpУ
6model/sequential_7/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput4model/sequential_7/conv2d_transpose_3/stack:output:0Mmodel/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0#model/concatenate/concat_2:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
28
6model/sequential_7/conv2d_transpose_3/conv2d_transposeю
<model/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpEmodel_sequential_7_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp™
-model/sequential_7/conv2d_transpose_3/BiasAddBiasAdd?model/sequential_7/conv2d_transpose_3/conv2d_transpose:output:0Dmodel/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2/
-model/sequential_7/conv2d_transpose_3/BiasAdd“
*model/sequential_7/conv2d_transpose_3/ReluRelu6model/sequential_7/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2,
*model/sequential_7/conv2d_transpose_3/ReluЉ
(model/sequential_7/up_sampling2d_3/ShapeShape8model/sequential_7/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2*
(model/sequential_7/up_sampling2d_3/ShapeЇ
6model/sequential_7/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model/sequential_7/up_sampling2d_3/strided_slice/stackЊ
8model/sequential_7/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/sequential_7/up_sampling2d_3/strided_slice/stack_1Њ
8model/sequential_7/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/sequential_7/up_sampling2d_3/strided_slice/stack_2†
0model/sequential_7/up_sampling2d_3/strided_sliceStridedSlice1model/sequential_7/up_sampling2d_3/Shape:output:0?model/sequential_7/up_sampling2d_3/strided_slice/stack:output:0Amodel/sequential_7/up_sampling2d_3/strided_slice/stack_1:output:0Amodel/sequential_7/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0model/sequential_7/up_sampling2d_3/strided_slice•
(model/sequential_7/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model/sequential_7/up_sampling2d_3/Constк
&model/sequential_7/up_sampling2d_3/mulMul9model/sequential_7/up_sampling2d_3/strided_slice:output:01model/sequential_7/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2(
&model/sequential_7/up_sampling2d_3/mul„
?model/sequential_7/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor8model/sequential_7/conv2d_transpose_3/Relu:activations:0*model/sequential_7/up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
half_pixel_centers(2A
?model/sequential_7/up_sampling2d_3/resize/ResizeNearestNeighborД
model/concatenate/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate/concat_3/axisО
model/concatenate/concat_3ConcatV2Pmodel/sequential_7/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0input_1(model/concatenate/concat_3/axis:output:0*
N*
T0*1
_output_shapes
:€€€€€€€€€ААA2
model/concatenate/concat_3У
model/conv2d_transpose_4/ShapeShape#model/concatenate/concat_3:output:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_4/Shape¶
,model/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_4/strided_slice/stack™
.model/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_4/strided_slice/stack_1™
.model/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_4/strided_slice/stack_2ш
&model/conv2d_transpose_4/strided_sliceStridedSlice'model/conv2d_transpose_4/Shape:output:05model/conv2d_transpose_4/strided_slice/stack:output:07model/conv2d_transpose_4/strided_slice/stack_1:output:07model/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_4/strided_sliceЗ
 model/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :А2"
 model/conv2d_transpose_4/stack/1З
 model/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :А2"
 model/conv2d_transpose_4/stack/2Ж
 model/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_4/stack/3®
model/conv2d_transpose_4/stackPack/model/conv2d_transpose_4/strided_slice:output:0)model/conv2d_transpose_4/stack/1:output:0)model/conv2d_transpose_4/stack/2:output:0)model/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_4/stack™
.model/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_4/strided_slice_1/stackЃ
0model/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_4/strided_slice_1/stack_1Ѓ
0model/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_4/strided_slice_1/stack_2В
(model/conv2d_transpose_4/strided_slice_1StridedSlice'model/conv2d_transpose_4/stack:output:07model/conv2d_transpose_4/strided_slice_1/stack:output:09model/conv2d_transpose_4/strided_slice_1/stack_1:output:09model/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_4/strided_slice_1ю
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:A*
dtype02:
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpв
)model/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_4/stack:output:0@model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0#model/concatenate/concat_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingVALID*
strides
2+
)model/conv2d_transpose_4/conv2d_transpose„
/model/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model/conv2d_transpose_4/BiasAdd/ReadVariableOpш
 model/conv2d_transpose_4/BiasAddBiasAdd2model/conv2d_transpose_4/conv2d_transpose:output:07model/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2"
 model/conv2d_transpose_4/BiasAdd≠
model/conv2d_transpose_4/ReluRelu)model/conv2d_transpose_4/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
model/conv2d_transpose_4/Relu™	
IdentityIdentity+model/conv2d_transpose_4/Relu:activations:00^model/conv2d_transpose_4/BiasAdd/ReadVariableOp9^model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp/^model/sequential/conv2d/BiasAdd/ReadVariableOp.^model/sequential/conv2d/Conv2D/ReadVariableOp3^model/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2^model/sequential_1/conv2d_1/Conv2D/ReadVariableOp3^model/sequential_2/conv2d_2/BiasAdd/ReadVariableOp2^model/sequential_2/conv2d_2/Conv2D/ReadVariableOp3^model/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2^model/sequential_3/conv2d_3/Conv2D/ReadVariableOp;^model/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpD^model/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp=^model/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpF^model/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=^model/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpF^model/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp=^model/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpF^model/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2b
/model/conv2d_transpose_4/BiasAdd/ReadVariableOp/model/conv2d_transpose_4/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2`
.model/sequential/conv2d/BiasAdd/ReadVariableOp.model/sequential/conv2d/BiasAdd/ReadVariableOp2^
-model/sequential/conv2d/Conv2D/ReadVariableOp-model/sequential/conv2d/Conv2D/ReadVariableOp2h
2model/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2model/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2f
1model/sequential_1/conv2d_1/Conv2D/ReadVariableOp1model/sequential_1/conv2d_1/Conv2D/ReadVariableOp2h
2model/sequential_2/conv2d_2/BiasAdd/ReadVariableOp2model/sequential_2/conv2d_2/BiasAdd/ReadVariableOp2f
1model/sequential_2/conv2d_2/Conv2D/ReadVariableOp1model/sequential_2/conv2d_2/Conv2D/ReadVariableOp2h
2model/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2model/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2f
1model/sequential_3/conv2d_3/Conv2D/ReadVariableOp1model/sequential_3/conv2d_3/Conv2D/ReadVariableOp2x
:model/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp:model/sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp2К
Cmodel/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpCmodel/sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp2|
<model/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp<model/sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp2О
Emodel/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpEmodel/sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2|
<model/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp<model/sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp2О
Emodel/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpEmodel/sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2|
<model/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp<model/sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp2О
Emodel/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpEmodel/sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1
щ
}
(__inference_conv2d_2_layer_call_fn_10508

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_82152
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
Ч
ґ
F__inference_sequential_1_layer_call_and_return_conditional_losses_8139
conv2d_1_input
conv2d_1_8132
conv2d_1_8134
identityИҐ conv2d_1/StatefulPartitionedCallЮ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_8132conv2d_1_8134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_81212"
 conv2d_1/StatefulPartitionedCallТ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_81002!
max_pooling2d_1/PartitionedCallІ
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€@@@
(
_user_specified_nameconv2d_1_input
Щ
И
+__inference_sequential_3_layer_call_fn_8357
conv2d_3_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83502
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_nameconv2d_3_input
’
Р
+__inference_sequential_4_layer_call_fn_8500
conv2d_transpose_input
unknown
	unknown_0
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84932
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:g c
/
_output_shapes
:€€€€€€€€€
0
_user_specified_nameconv2d_transpose_input
Щ
И
+__inference_sequential_1_layer_call_fn_8169
conv2d_1_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81622
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€@@@
(
_user_specified_nameconv2d_1_input
ц
W
+__inference_concatenate_layer_call_fn_10189
inputs_0
inputs_1
identityЏ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ААA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91592
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААA2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:€€€€€€€€€АА:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:€€€€€€€€€АА
"
_user_specified_name
inputs/1
г

¶
D__inference_sequential_layer_call_and_return_conditional_losses_8087

inputs
conv2d_8080
conv2d_8082
identityИҐconv2d/StatefulPartitionedCallО
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8080conv2d_8082*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_80272 
conv2d/StatefulPartitionedCallК
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_80062
max_pooling2d/PartitionedCall£
IdentityIdentity&max_pooling2d/PartitionedCall:output:0^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ч
ґ
F__inference_sequential_1_layer_call_and_return_conditional_losses_8149
conv2d_1_input
conv2d_1_8142
conv2d_1_8144
identityИҐ conv2d_1/StatefulPartitionedCallЮ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_8142conv2d_1_8144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_81212"
 conv2d_1/StatefulPartitionedCallТ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_81002!
max_pooling2d_1/PartitionedCallІ
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€@@@
(
_user_specified_nameconv2d_1_input
П
r
F__inference_concatenate_layer_call_and_return_conditional_losses_10157
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЙ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@`2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@`2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :€€€€€€€€€@@@:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€@@@
"
_user_specified_name
inputs/1
г
ё
F__inference_sequential_5_layer_call_and_return_conditional_losses_8585
conv2d_transpose_1_input
conv2d_transpose_1_8578
conv2d_transpose_1_8580
identityИҐ*conv2d_transpose_1/StatefulPartitionedCallм
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_1_inputconv2d_transpose_1_8578conv2d_transpose_1_8580*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_85352,
*conv2d_transpose_1/StatefulPartitionedCallЃ
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_85582!
up_sampling2d_1/PartitionedCall√
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€
2
_user_specified_nameconv2d_transpose_1_input
Б
А
+__inference_sequential_1_layer_call_fn_9984

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81812
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
Ж
o
E__inference_concatenate_layer_call_and_return_conditional_losses_9085

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЗ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€  02
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:€€€€€€€€€  02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€   :i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
¶
Б
,__inference_sequential_4_layer_call_fn_10141

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84742
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
©
J
.__inference_max_pooling2d_1_layer_call_fn_8106

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_81002
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А
~
)__inference_sequential_layer_call_fn_9942

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
¶
Б
,__inference_sequential_5_layer_call_fn_10284

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_86172
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ђ
с
$__inference_model_layer_call_fn_9321
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_92822
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1
П
r
F__inference_concatenate_layer_call_and_return_conditional_losses_10170
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЙ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€  02
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:€€€€€€€€€  02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€   :k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€   
"
_user_specified_name
inputs/1
≤.
ј
G__inference_sequential_7_layer_call_and_return_conditional_losses_10430

inputs?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_3/BiasAdd/ReadVariableOpҐ2conv2d_transpose_3/conv2d_transpose/ReadVariableOpj
conv2d_transpose_3/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_3/ShapeЪ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackЮ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1Ю
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2‘
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/3Д
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackЮ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stackҐ
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1Ґ
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2ё
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1м
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp™
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose≈
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpё
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d_transpose_3/BiasAddЩ
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d_transpose_3/ReluГ
up_sampling2d_3/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/ShapeФ
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stackШ
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1Ш
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2Ѓ
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/ConstЮ
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulЛ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_3/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighborь
IdentityIdentity=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@`
 
_user_specified_nameinputs
Ѓ.
ј
G__inference_sequential_5_layer_call_and_return_conditional_losses_10234

inputs?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_1/BiasAdd/ReadVariableOpҐ2conv2d_transpose_1/conv2d_transpose/ReadVariableOpj
conv2d_transpose_1/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_1/ShapeЪ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackЮ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1Ю
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2‘
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3Д
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackЮ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackҐ
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1Ґ
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2ё
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1м
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp™
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose≈
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpё
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose_1/BiasAddЩ
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose_1/ReluГ
up_sampling2d_1/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/ShapeФ
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stackШ
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1Ш
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2Ѓ
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/ConstЮ
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulЙ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_1/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€  *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborъ
IdentityIdentity=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А
~
)__inference_sequential_layer_call_fn_9933

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80682
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ѓ.
ј
G__inference_sequential_5_layer_call_and_return_conditional_losses_10266

inputs?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_1/BiasAdd/ReadVariableOpҐ2conv2d_transpose_1/conv2d_transpose/ReadVariableOpj
conv2d_transpose_1/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_1/ShapeЪ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackЮ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1Ю
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2‘
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/3Д
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackЮ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackҐ
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1Ґ
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2ё
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1м
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp™
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose≈
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpё
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose_1/BiasAddЩ
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose_1/ReluГ
up_sampling2d_1/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/ShapeФ
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stackШ
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1Ш
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2Ѓ
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/ConstЮ
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulЙ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_1/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€  *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighborъ
IdentityIdentity=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ћ

џ
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8309

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ.
ј
G__inference_sequential_6_layer_call_and_return_conditional_losses_10316

inputs?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_2/BiasAdd/ReadVariableOpҐ2conv2d_transpose_2/conv2d_transpose/ReadVariableOpj
conv2d_transpose_2/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_2/ShapeЪ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackЮ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1Ю
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2‘
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/3Д
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackЮ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackҐ
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1Ґ
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2ё
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1м
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp™
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose≈
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpё
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_transpose_2/BiasAddЩ
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_transpose_2/ReluГ
up_sampling2d_2/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/ShapeФ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackШ
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1Ш
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2Ѓ
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/ConstЮ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulЙ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_2/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighborъ
IdentityIdentity=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  0
 
_user_specified_nameinputs
Ж
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8558

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ќ

№
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10479

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
љЄ
†
?__inference_model_layer_call_and_return_conditional_losses_9642

inputs4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource8
4sequential_1_conv2d_1_conv2d_readvariableop_resource9
5sequential_1_conv2d_1_biasadd_readvariableop_resource8
4sequential_2_conv2d_2_conv2d_readvariableop_resource9
5sequential_2_conv2d_2_biasadd_readvariableop_resource8
4sequential_3_conv2d_3_conv2d_readvariableop_resource9
5sequential_3_conv2d_3_biasadd_readvariableop_resourceJ
Fsequential_4_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_4_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_5_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_5_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_6_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_6_conv2d_transpose_2_biasadd_readvariableop_resourceL
Hsequential_7_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceC
?sequential_7_conv2d_transpose_3_biasadd_readvariableop_resource?
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_4_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_4/BiasAdd/ReadVariableOpҐ2conv2d_transpose_4/conv2d_transpose/ReadVariableOpҐ(sequential/conv2d/BiasAdd/ReadVariableOpҐ'sequential/conv2d/Conv2D/ReadVariableOpҐ,sequential_1/conv2d_1/BiasAdd/ReadVariableOpҐ+sequential_1/conv2d_1/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_2/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_2/Conv2D/ReadVariableOpҐ,sequential_3/conv2d_3/BiasAdd/ReadVariableOpҐ+sequential_3/conv2d_3/Conv2D/ReadVariableOpҐ4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpҐ=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpҐ6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpҐ?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpҐ?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpҐ?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpЋ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpџ
sequential/conv2d/Conv2DConv2Dinputs/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2
sequential/conv2d/Conv2D¬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp“
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
sequential/conv2d/BiasAddШ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
sequential/conv2d/Reluв
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool„
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOpИ
sequential_1/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2Dќ
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpа
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
sequential_1/conv2d_1/BiasAddҐ
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
sequential_1/conv2d_1/Reluо
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_1/MaxPool„
+sequential_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_2/Conv2D/ReadVariableOpМ
sequential_2/conv2d_2/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:03sequential_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2
sequential_2/conv2d_2/Conv2Dќ
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpа
sequential_2/conv2d_2/BiasAddBiasAdd%sequential_2/conv2d_2/Conv2D:output:04sequential_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
sequential_2/conv2d_2/BiasAddҐ
sequential_2/conv2d_2/ReluRelu&sequential_2/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2
sequential_2/conv2d_2/Reluо
$sequential_2/max_pooling2d_2/MaxPoolMaxPool(sequential_2/conv2d_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_2/MaxPool„
+sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_3/conv2d_3/Conv2D/ReadVariableOpМ
sequential_3/conv2d_3/Conv2DConv2D-sequential_2/max_pooling2d_2/MaxPool:output:03sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2
sequential_3/conv2d_3/Conv2Dќ
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpа
sequential_3/conv2d_3/BiasAddBiasAdd%sequential_3/conv2d_3/Conv2D:output:04sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential_3/conv2d_3/BiasAddҐ
sequential_3/conv2d_3/ReluRelu&sequential_3/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential_3/conv2d_3/Reluо
$sequential_3/max_pooling2d_3/MaxPoolMaxPool(sequential_3/conv2d_3/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_3/MaxPoolІ
#sequential_4/conv2d_transpose/ShapeShape-sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2%
#sequential_4/conv2d_transpose/Shape∞
1sequential_4/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_4/conv2d_transpose/strided_slice/stackі
3sequential_4/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_4/conv2d_transpose/strided_slice/stack_1і
3sequential_4/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_4/conv2d_transpose/strided_slice/stack_2Ц
+sequential_4/conv2d_transpose/strided_sliceStridedSlice,sequential_4/conv2d_transpose/Shape:output:0:sequential_4/conv2d_transpose/strided_slice/stack:output:0<sequential_4/conv2d_transpose/strided_slice/stack_1:output:0<sequential_4/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_4/conv2d_transpose/strided_sliceР
%sequential_4/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_4/conv2d_transpose/stack/1Р
%sequential_4/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_4/conv2d_transpose/stack/2Р
%sequential_4/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_4/conv2d_transpose/stack/3∆
#sequential_4/conv2d_transpose/stackPack4sequential_4/conv2d_transpose/strided_slice:output:0.sequential_4/conv2d_transpose/stack/1:output:0.sequential_4/conv2d_transpose/stack/2:output:0.sequential_4/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_4/conv2d_transpose/stackі
3sequential_4/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_4/conv2d_transpose/strided_slice_1/stackЄ
5sequential_4/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_4/conv2d_transpose/strided_slice_1/stack_1Є
5sequential_4/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_4/conv2d_transpose/strided_slice_1/stack_2†
-sequential_4/conv2d_transpose/strided_slice_1StridedSlice,sequential_4/conv2d_transpose/stack:output:0<sequential_4/conv2d_transpose/strided_slice_1/stack:output:0>sequential_4/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_4/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_4/conv2d_transpose/strided_slice_1Н
=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_4_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02?
=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpэ
.sequential_4/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_4/conv2d_transpose/stack:output:0Esequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0-sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
20
.sequential_4/conv2d_transpose/conv2d_transposeж
4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_4_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpК
%sequential_4/conv2d_transpose/BiasAddBiasAdd7sequential_4/conv2d_transpose/conv2d_transpose:output:0<sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%sequential_4/conv2d_transpose/BiasAddЇ
"sequential_4/conv2d_transpose/ReluRelu.sequential_4/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2$
"sequential_4/conv2d_transpose/Relu§
 sequential_4/up_sampling2d/ShapeShape0sequential_4/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 sequential_4/up_sampling2d/Shape™
.sequential_4/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_4/up_sampling2d/strided_slice/stackЃ
0sequential_4/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_4/up_sampling2d/strided_slice/stack_1Ѓ
0sequential_4/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_4/up_sampling2d/strided_slice/stack_2р
(sequential_4/up_sampling2d/strided_sliceStridedSlice)sequential_4/up_sampling2d/Shape:output:07sequential_4/up_sampling2d/strided_slice/stack:output:09sequential_4/up_sampling2d/strided_slice/stack_1:output:09sequential_4/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(sequential_4/up_sampling2d/strided_sliceХ
 sequential_4/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2"
 sequential_4/up_sampling2d/Const 
sequential_4/up_sampling2d/mulMul1sequential_4/up_sampling2d/strided_slice:output:0)sequential_4/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2 
sequential_4/up_sampling2d/mulµ
7sequential_4/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor0sequential_4/conv2d_transpose/Relu:activations:0"sequential_4/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€*
half_pixel_centers(29
7sequential_4/up_sampling2d/resize/ResizeNearestNeighbort
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisТ
concatenate/concatConcatV2Hsequential_4/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0-sequential_2/max_pooling2d_2/MaxPool:output:0 concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€2
concatenate/concatЩ
%sequential_5/conv2d_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_1/Shapeі
3sequential_5/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_1/strided_slice/stackЄ
5sequential_5/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_1/strided_slice/stack_1Є
5sequential_5/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_1/strided_slice/stack_2Ґ
-sequential_5/conv2d_transpose_1/strided_sliceStridedSlice.sequential_5/conv2d_transpose_1/Shape:output:0<sequential_5/conv2d_transpose_1/strided_slice/stack:output:0>sequential_5/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_1/strided_sliceФ
'sequential_5/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_1/stack/1Ф
'sequential_5/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_1/stack/2Ф
'sequential_5/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_1/stack/3“
%sequential_5/conv2d_transpose_1/stackPack6sequential_5/conv2d_transpose_1/strided_slice:output:00sequential_5/conv2d_transpose_1/stack/1:output:00sequential_5/conv2d_transpose_1/stack/2:output:00sequential_5/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_1/stackЄ
5sequential_5/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_1/strided_slice_1/stackЉ
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_1Љ
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_2ђ
/sequential_5/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_1/stack:output:0>sequential_5/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_1/strided_slice_1У
?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02A
?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpу
0sequential_5/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_1/stack:output:0Gsequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_1/conv2d_transposeм
6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpТ
'sequential_5/conv2d_transpose_1/BiasAddBiasAdd9sequential_5/conv2d_transpose_1/conv2d_transpose:output:0>sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'sequential_5/conv2d_transpose_1/BiasAddј
$sequential_5/conv2d_transpose_1/ReluRelu0sequential_5/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2&
$sequential_5/conv2d_transpose_1/Relu™
"sequential_5/up_sampling2d_1/ShapeShape2sequential_5/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential_5/up_sampling2d_1/ShapeЃ
0sequential_5/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_5/up_sampling2d_1/strided_slice/stack≤
2sequential_5/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_5/up_sampling2d_1/strided_slice/stack_1≤
2sequential_5/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_5/up_sampling2d_1/strided_slice/stack_2ь
*sequential_5/up_sampling2d_1/strided_sliceStridedSlice+sequential_5/up_sampling2d_1/Shape:output:09sequential_5/up_sampling2d_1/strided_slice/stack:output:0;sequential_5/up_sampling2d_1/strided_slice/stack_1:output:0;sequential_5/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_5/up_sampling2d_1/strided_sliceЩ
"sequential_5/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_5/up_sampling2d_1/Const“
 sequential_5/up_sampling2d_1/mulMul3sequential_5/up_sampling2d_1/strided_slice:output:0+sequential_5/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 sequential_5/up_sampling2d_1/mulљ
9sequential_5/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_5/conv2d_transpose_1/Relu:activations:0$sequential_5/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€  *
half_pixel_centers(2;
9sequential_5/up_sampling2d_1/resize/ResizeNearestNeighborx
concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat_1/axisЪ
concatenate/concat_1ConcatV2Jsequential_5/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0-sequential_1/max_pooling2d_1/MaxPool:output:0"concatenate/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€  02
concatenate/concat_1Ы
%sequential_6/conv2d_transpose_2/ShapeShapeconcatenate/concat_1:output:0*
T0*
_output_shapes
:2'
%sequential_6/conv2d_transpose_2/Shapeі
3sequential_6/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_6/conv2d_transpose_2/strided_slice/stackЄ
5sequential_6/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_6/conv2d_transpose_2/strided_slice/stack_1Є
5sequential_6/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_6/conv2d_transpose_2/strided_slice/stack_2Ґ
-sequential_6/conv2d_transpose_2/strided_sliceStridedSlice.sequential_6/conv2d_transpose_2/Shape:output:0<sequential_6/conv2d_transpose_2/strided_slice/stack:output:0>sequential_6/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_6/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_6/conv2d_transpose_2/strided_sliceФ
'sequential_6/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/conv2d_transpose_2/stack/1Ф
'sequential_6/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/conv2d_transpose_2/stack/2Ф
'sequential_6/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/conv2d_transpose_2/stack/3“
%sequential_6/conv2d_transpose_2/stackPack6sequential_6/conv2d_transpose_2/strided_slice:output:00sequential_6/conv2d_transpose_2/stack/1:output:00sequential_6/conv2d_transpose_2/stack/2:output:00sequential_6/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/conv2d_transpose_2/stackЄ
5sequential_6/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_6/conv2d_transpose_2/strided_slice_1/stackЉ
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_1Љ
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_2ђ
/sequential_6/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_6/conv2d_transpose_2/stack:output:0>sequential_6/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_6/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_6/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_6/conv2d_transpose_2/strided_slice_1У
?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_6_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02A
?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpх
0sequential_6/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_6/conv2d_transpose_2/stack:output:0Gsequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0concatenate/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
22
0sequential_6/conv2d_transpose_2/conv2d_transposeм
6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_6_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpТ
'sequential_6/conv2d_transpose_2/BiasAddBiasAdd9sequential_6/conv2d_transpose_2/conv2d_transpose:output:0>sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2)
'sequential_6/conv2d_transpose_2/BiasAddј
$sequential_6/conv2d_transpose_2/ReluRelu0sequential_6/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2&
$sequential_6/conv2d_transpose_2/Relu™
"sequential_6/up_sampling2d_2/ShapeShape2sequential_6/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential_6/up_sampling2d_2/ShapeЃ
0sequential_6/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_6/up_sampling2d_2/strided_slice/stack≤
2sequential_6/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_6/up_sampling2d_2/strided_slice/stack_1≤
2sequential_6/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_6/up_sampling2d_2/strided_slice/stack_2ь
*sequential_6/up_sampling2d_2/strided_sliceStridedSlice+sequential_6/up_sampling2d_2/Shape:output:09sequential_6/up_sampling2d_2/strided_slice/stack:output:0;sequential_6/up_sampling2d_2/strided_slice/stack_1:output:0;sequential_6/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_6/up_sampling2d_2/strided_sliceЩ
"sequential_6/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_6/up_sampling2d_2/Const“
 sequential_6/up_sampling2d_2/mulMul3sequential_6/up_sampling2d_2/strided_slice:output:0+sequential_6/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2"
 sequential_6/up_sampling2d_2/mulљ
9sequential_6/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_6/conv2d_transpose_2/Relu:activations:0$sequential_6/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
half_pixel_centers(2;
9sequential_6/up_sampling2d_2/resize/ResizeNearestNeighborx
concatenate/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat_2/axisЦ
concatenate/concat_2ConcatV2Jsequential_6/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0)sequential/max_pooling2d/MaxPool:output:0"concatenate/concat_2/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@`2
concatenate/concat_2Ы
%sequential_7/conv2d_transpose_3/ShapeShapeconcatenate/concat_2:output:0*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_3/Shapeі
3sequential_7/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_7/conv2d_transpose_3/strided_slice/stackЄ
5sequential_7/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_3/strided_slice/stack_1Є
5sequential_7/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_3/strided_slice/stack_2Ґ
-sequential_7/conv2d_transpose_3/strided_sliceStridedSlice.sequential_7/conv2d_transpose_3/Shape:output:0<sequential_7/conv2d_transpose_3/strided_slice/stack:output:0>sequential_7/conv2d_transpose_3/strided_slice/stack_1:output:0>sequential_7/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_7/conv2d_transpose_3/strided_sliceФ
'sequential_7/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_3/stack/1Ф
'sequential_7/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_3/stack/2Ф
'sequential_7/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_3/stack/3“
%sequential_7/conv2d_transpose_3/stackPack6sequential_7/conv2d_transpose_3/strided_slice:output:00sequential_7/conv2d_transpose_3/stack/1:output:00sequential_7/conv2d_transpose_3/stack/2:output:00sequential_7/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_3/stackЄ
5sequential_7/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_7/conv2d_transpose_3/strided_slice_1/stackЉ
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_1Љ
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_2ђ
/sequential_7/conv2d_transpose_3/strided_slice_1StridedSlice.sequential_7/conv2d_transpose_3/stack:output:0>sequential_7/conv2d_transpose_3/strided_slice_1/stack:output:0@sequential_7/conv2d_transpose_3/strided_slice_1/stack_1:output:0@sequential_7/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_7/conv2d_transpose_3/strided_slice_1У
?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_7_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype02A
?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpх
0sequential_7/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.sequential_7/conv2d_transpose_3/stack:output:0Gsequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0concatenate/concat_2:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
22
0sequential_7/conv2d_transpose_3/conv2d_transposeм
6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_7_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpТ
'sequential_7/conv2d_transpose_3/BiasAddBiasAdd9sequential_7/conv2d_transpose_3/conv2d_transpose:output:0>sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2)
'sequential_7/conv2d_transpose_3/BiasAddј
$sequential_7/conv2d_transpose_3/ReluRelu0sequential_7/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2&
$sequential_7/conv2d_transpose_3/Relu™
"sequential_7/up_sampling2d_3/ShapeShape2sequential_7/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential_7/up_sampling2d_3/ShapeЃ
0sequential_7/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_7/up_sampling2d_3/strided_slice/stack≤
2sequential_7/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_7/up_sampling2d_3/strided_slice/stack_1≤
2sequential_7/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_7/up_sampling2d_3/strided_slice/stack_2ь
*sequential_7/up_sampling2d_3/strided_sliceStridedSlice+sequential_7/up_sampling2d_3/Shape:output:09sequential_7/up_sampling2d_3/strided_slice/stack:output:0;sequential_7/up_sampling2d_3/strided_slice/stack_1:output:0;sequential_7/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_7/up_sampling2d_3/strided_sliceЩ
"sequential_7/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_7/up_sampling2d_3/Const“
 sequential_7/up_sampling2d_3/mulMul3sequential_7/up_sampling2d_3/strided_slice:output:0+sequential_7/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2"
 sequential_7/up_sampling2d_3/mulњ
9sequential_7/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_7/conv2d_transpose_3/Relu:activations:0$sequential_7/up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
half_pixel_centers(2;
9sequential_7/up_sampling2d_3/resize/ResizeNearestNeighborx
concatenate/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat_3/axisх
concatenate/concat_3ConcatV2Jsequential_7/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0inputs"concatenate/concat_3/axis:output:0*
N*
T0*1
_output_shapes
:€€€€€€€€€ААA2
concatenate/concat_3Б
conv2d_transpose_4/ShapeShapeconcatenate/concat_3:output:0*
T0*
_output_shapes
:2
conv2d_transpose_4/ShapeЪ
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stackЮ
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1Ю
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2‘
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice{
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :А2
conv2d_transpose_4/stack/1{
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :А2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3Д
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stackЮ
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stackҐ
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1Ґ
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2ё
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1м
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:A*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpƒ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0concatenate/concat_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingVALID*
strides
2%
#conv2d_transpose_4/conv2d_transpose≈
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOpа
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_transpose_4/BiasAddЫ
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_transpose_4/ReluЄ
IdentityIdentity%conv2d_transpose_4/Relu:activations:0*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp-^sequential_2/conv2d_2/BiasAdd/ReadVariableOp,^sequential_2/conv2d_2/Conv2D/ReadVariableOp-^sequential_3/conv2d_3/BiasAdd/ReadVariableOp,^sequential_3/conv2d_3/Conv2D/ReadVariableOp5^sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp@^sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_2/BiasAdd/ReadVariableOp,sequential_2/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_2/Conv2D/ReadVariableOp+sequential_2/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_3/BiasAdd/ReadVariableOp,sequential_3/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_3/Conv2D/ReadVariableOp+sequential_3/conv2d_3/Conv2D/ReadVariableOp2l
4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp2В
?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp2В
?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp2В
?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ЙЦ
Л$
!__inference__traced_restore_10957
file_prefix.
*assignvariableop_conv2d_transpose_4_kernel.
*assignvariableop_1_conv2d_transpose_4_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate$
 assignvariableop_7_conv2d_kernel"
assignvariableop_8_conv2d_bias&
"assignvariableop_9_conv2d_1_kernel%
!assignvariableop_10_conv2d_1_bias'
#assignvariableop_11_conv2d_2_kernel%
!assignvariableop_12_conv2d_2_bias'
#assignvariableop_13_conv2d_3_kernel%
!assignvariableop_14_conv2d_3_bias/
+assignvariableop_15_conv2d_transpose_kernel-
)assignvariableop_16_conv2d_transpose_bias1
-assignvariableop_17_conv2d_transpose_1_kernel/
+assignvariableop_18_conv2d_transpose_1_bias1
-assignvariableop_19_conv2d_transpose_2_kernel/
+assignvariableop_20_conv2d_transpose_2_bias1
-assignvariableop_21_conv2d_transpose_3_kernel/
+assignvariableop_22_conv2d_transpose_3_bias
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1
assignvariableop_27_total_2
assignvariableop_28_count_2.
*assignvariableop_29_total_confusion_matrix8
4assignvariableop_30_adam_conv2d_transpose_4_kernel_m6
2assignvariableop_31_adam_conv2d_transpose_4_bias_m,
(assignvariableop_32_adam_conv2d_kernel_m*
&assignvariableop_33_adam_conv2d_bias_m.
*assignvariableop_34_adam_conv2d_1_kernel_m,
(assignvariableop_35_adam_conv2d_1_bias_m.
*assignvariableop_36_adam_conv2d_2_kernel_m,
(assignvariableop_37_adam_conv2d_2_bias_m.
*assignvariableop_38_adam_conv2d_3_kernel_m,
(assignvariableop_39_adam_conv2d_3_bias_m6
2assignvariableop_40_adam_conv2d_transpose_kernel_m4
0assignvariableop_41_adam_conv2d_transpose_bias_m8
4assignvariableop_42_adam_conv2d_transpose_1_kernel_m6
2assignvariableop_43_adam_conv2d_transpose_1_bias_m8
4assignvariableop_44_adam_conv2d_transpose_2_kernel_m6
2assignvariableop_45_adam_conv2d_transpose_2_bias_m8
4assignvariableop_46_adam_conv2d_transpose_3_kernel_m6
2assignvariableop_47_adam_conv2d_transpose_3_bias_m8
4assignvariableop_48_adam_conv2d_transpose_4_kernel_v6
2assignvariableop_49_adam_conv2d_transpose_4_bias_v,
(assignvariableop_50_adam_conv2d_kernel_v*
&assignvariableop_51_adam_conv2d_bias_v.
*assignvariableop_52_adam_conv2d_1_kernel_v,
(assignvariableop_53_adam_conv2d_1_bias_v.
*assignvariableop_54_adam_conv2d_2_kernel_v,
(assignvariableop_55_adam_conv2d_2_bias_v.
*assignvariableop_56_adam_conv2d_3_kernel_v,
(assignvariableop_57_adam_conv2d_3_bias_v6
2assignvariableop_58_adam_conv2d_transpose_kernel_v4
0assignvariableop_59_adam_conv2d_transpose_bias_v8
4assignvariableop_60_adam_conv2d_transpose_1_kernel_v6
2assignvariableop_61_adam_conv2d_transpose_1_bias_v8
4assignvariableop_62_adam_conv2d_transpose_2_kernel_v6
2assignvariableop_63_adam_conv2d_transpose_2_bias_v8
4assignvariableop_64_adam_conv2d_transpose_3_kernel_v6
2assignvariableop_65_adam_conv2d_transpose_3_bias_v
identity_67ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9√#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*ѕ"
value≈"B¬"CB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBEkeras_api/metrics/3/total_confusion_matrix/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЧ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesэ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ґ
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity©
AssignVariableOpAssignVariableOp*assignvariableop_conv2d_transpose_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ѓ
AssignVariableOp_1AssignVariableOp*assignvariableop_1_conv2d_transpose_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6™
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv2d_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9І
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ђ
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ђ
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15≥
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_conv2d_transpose_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp-assignvariableop_17_conv2d_transpose_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18≥
AssignVariableOp_18AssignVariableOp+assignvariableop_18_conv2d_transpose_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19µ
AssignVariableOp_19AssignVariableOp-assignvariableop_19_conv2d_transpose_2_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20≥
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_conv2d_transpose_3_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22≥
AssignVariableOp_22AssignVariableOp+assignvariableop_22_conv2d_transpose_3_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27£
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28£
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≤
AssignVariableOp_29AssignVariableOp*assignvariableop_29_total_confusion_matrixIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Љ
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_conv2d_transpose_4_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ї
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_conv2d_transpose_4_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32∞
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ѓ
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_conv2d_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34≤
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_1_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35∞
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36≤
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_2_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37∞
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_2_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38≤
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_conv2d_3_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39∞
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv2d_3_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ї
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_conv2d_transpose_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Є
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_conv2d_transpose_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Љ
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_conv2d_transpose_1_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ї
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_conv2d_transpose_1_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Љ
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_conv2d_transpose_2_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ї
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adam_conv2d_transpose_2_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Љ
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_conv2d_transpose_3_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ї
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_conv2d_transpose_3_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Љ
AssignVariableOp_48AssignVariableOp4assignvariableop_48_adam_conv2d_transpose_4_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ї
AssignVariableOp_49AssignVariableOp2assignvariableop_49_adam_conv2d_transpose_4_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50∞
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ѓ
AssignVariableOp_51AssignVariableOp&assignvariableop_51_adam_conv2d_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52≤
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_conv2d_1_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53∞
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv2d_1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54≤
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_2_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55∞
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv2d_2_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56≤
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_3_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57∞
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_conv2d_3_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ї
AssignVariableOp_58AssignVariableOp2assignvariableop_58_adam_conv2d_transpose_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Є
AssignVariableOp_59AssignVariableOp0assignvariableop_59_adam_conv2d_transpose_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Љ
AssignVariableOp_60AssignVariableOp4assignvariableop_60_adam_conv2d_transpose_1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ї
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_conv2d_transpose_1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Љ
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_conv2d_transpose_2_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ї
AssignVariableOp_63AssignVariableOp2assignvariableop_63_adam_conv2d_transpose_2_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Љ
AssignVariableOp_64AssignVariableOp4assignvariableop_64_adam_conv2d_transpose_3_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ї
AssignVariableOp_65AssignVariableOp2assignvariableop_65_adam_conv2d_transpose_3_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_659
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpК
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_66э
Identity_67IdentityIdentity_66:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_67"#
identity_67Identity_67:output:0*Я
_input_shapesН
К: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ђ
с
$__inference_model_layer_call_fn_9415
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_93762
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1
©
р
$__inference_model_layer_call_fn_9859

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_92822
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ж
e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8806

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ќ

№
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10519

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
©
J
.__inference_max_pooling2d_2_layer_call_fn_8200

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_81942
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€

Ѓ
F__inference_sequential_2_layer_call_and_return_conditional_losses_8275

inputs
conv2d_2_8268
conv2d_2_8270
identityИҐ conv2d_2/StatefulPartitionedCallЦ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_8268conv2d_2_8270*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_82152"
 conv2d_2/StatefulPartitionedCallТ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_81942!
max_pooling2d_2/PartitionedCallІ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
ѓ'
щ
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_8911

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:A*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€A::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€A
 
_user_specified_nameinputs
€

Ѓ
F__inference_sequential_3_layer_call_and_return_conditional_losses_8369

inputs
conv2d_3_8362
conv2d_3_8364
identityИҐ conv2d_3/StatefulPartitionedCallЦ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_8362conv2d_3_8364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_83092"
 conv2d_3/StatefulPartitionedCallТ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_82882!
max_pooling2d_3/PartitionedCallІ
IdentityIdentity(max_pooling2d_3/PartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
€
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8194

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
И
+__inference_sequential_3_layer_call_fn_8376
conv2d_3_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83692
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_nameconv2d_3_input
џ
Т
+__inference_sequential_7_layer_call_fn_8853
conv2d_transpose_3_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88462
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€@@`
2
_user_specified_nameconv2d_transpose_3_input
’
Ж
1__inference_conv2d_transpose_4_layer_call_fn_8921

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_89112
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€A::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€A
 
_user_specified_nameinputs
ч,
Є
G__inference_sequential_4_layer_call_and_return_conditional_losses_10100

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identityИҐ'conv2d_transpose/BiasAdd/ReadVariableOpҐ0conv2d_transpose/conv2d_transpose/ReadVariableOpf
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose/ShapeЦ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackЪ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1Ъ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2»
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3ш
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackЪ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackЮ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1Ю
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2“
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1ж
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpҐ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transposeњ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp÷
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose/BiasAddУ
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose/Relu}
up_sampling2d/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/ShapeР
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stackФ
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1Ф
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2Ґ
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/ConstЦ
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulБ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#conv2d_transpose/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborф
IdentityIdentity;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
г
ё
F__inference_sequential_5_layer_call_and_return_conditional_losses_8575
conv2d_transpose_1_input
conv2d_transpose_1_8568
conv2d_transpose_1_8570
identityИҐ*conv2d_transpose_1/StatefulPartitionedCallм
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_1_inputconv2d_transpose_1_8568conv2d_transpose_1_8570*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_85352,
*conv2d_transpose_1/StatefulPartitionedCallЃ
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_85582!
up_sampling2d_1/PartitionedCall√
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€
2
_user_specified_nameconv2d_transpose_1_input
≠
ћ
F__inference_sequential_5_layer_call_and_return_conditional_losses_8617

inputs
conv2d_transpose_1_8610
conv2d_transpose_1_8612
identityИҐ*conv2d_transpose_1/StatefulPartitionedCallЏ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_1_8610conv2d_transpose_1_8612*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_85352,
*conv2d_transpose_1/StatefulPartitionedCallЃ
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_85582!
up_sampling2d_1/PartitionedCall√
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
Ж
1__inference_conv2d_transpose_1_layer_call_fn_8545

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_85352
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѓ.
ј
G__inference_sequential_6_layer_call_and_return_conditional_losses_10348

inputs?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_2/BiasAdd/ReadVariableOpҐ2conv2d_transpose_2/conv2d_transpose/ReadVariableOpj
conv2d_transpose_2/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_2/ShapeЪ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackЮ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1Ю
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2‘
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/3Д
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackЮ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackҐ
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1Ґ
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2ё
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1м
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp™
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose≈
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpё
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_transpose_2/BiasAddЩ
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_transpose_2/ReluГ
up_sampling2d_2/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/ShapeФ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackШ
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1Ш
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2Ѓ
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/ConstЮ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulЙ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_2/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighborъ
IdentityIdentity=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  0
 
_user_specified_nameinputs
Ч
ґ
F__inference_sequential_2_layer_call_and_return_conditional_losses_8233
conv2d_2_input
conv2d_2_8226
conv2d_2_8228
identityИҐ conv2d_2/StatefulPartitionedCallЮ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_8226conv2d_2_8228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_82152"
 conv2d_2/StatefulPartitionedCallТ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_81942!
max_pooling2d_2/PartitionedCallІ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€   
(
_user_specified_nameconv2d_2_input
¶
Б
,__inference_sequential_6_layer_call_fn_10366

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87412
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  0
 
_user_specified_nameinputs
џ
Т
+__inference_sequential_6_layer_call_fn_8729
conv2d_transpose_2_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87222
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€  0
2
_user_specified_nameconv2d_transpose_2_input
„

Џ
A__inference_conv2d_layer_call_and_return_conditional_losses_10459

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
Relu°
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
љ
÷
F__inference_sequential_4_layer_call_and_return_conditional_losses_8461
conv2d_transpose_input
conv2d_transpose_8454
conv2d_transpose_8456
identityИҐ(conv2d_transpose/StatefulPartitionedCallа
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputconv2d_transpose_8454conv2d_transpose_8456*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84112*
(conv2d_transpose/StatefulPartitionedCall¶
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_84342
up_sampling2d/PartitionedCallњ
IdentityIdentity&up_sampling2d/PartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:g c
/
_output_shapes
:€€€€€€€€€
0
_user_specified_nameconv2d_transpose_input
о
W
+__inference_concatenate_layer_call_fn_10176
inputs_0
inputs_1
identityЎ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90852
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€  02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€   :k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€   
"
_user_specified_name
inputs/1
≤.
ј
G__inference_sequential_7_layer_call_and_return_conditional_losses_10398

inputs?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_3/BiasAdd/ReadVariableOpҐ2conv2d_transpose_3/conv2d_transpose/ReadVariableOpj
conv2d_transpose_3/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose_3/ShapeЪ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackЮ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1Ю
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2‘
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/3Д
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackЮ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stackҐ
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1Ґ
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2ё
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1м
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp™
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose≈
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpё
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d_transpose_3/BiasAddЩ
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2
conv2d_transpose_3/ReluГ
up_sampling2d_3/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/ShapeФ
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stackШ
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1Ш
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2Ѓ
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/ConstЮ
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulЛ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%conv2d_transpose_3/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighborь
IdentityIdentity=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@`
 
_user_specified_nameinputs
•
H
,__inference_up_sampling2d_layer_call_fn_8440

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_84342
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
÷

ў
@__inference_conv2d_layer_call_and_return_conditional_losses_8027

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
Relu°
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Д
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8434

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
}
(__inference_conv2d_1_layer_call_fn_10488

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_81212
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
Щ
И
+__inference_sequential_2_layer_call_fn_8282
conv2d_2_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€   
(
_user_specified_nameconv2d_2_input
љ
÷
F__inference_sequential_4_layer_call_and_return_conditional_losses_8451
conv2d_transpose_input
conv2d_transpose_8444
conv2d_transpose_8446
identityИҐ(conv2d_transpose/StatefulPartitionedCallа
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputconv2d_transpose_8444conv2d_transpose_8446*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84112*
(conv2d_transpose/StatefulPartitionedCall¶
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_84342
up_sampling2d/PartitionedCallњ
IdentityIdentity&up_sampling2d/PartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:g c
/
_output_shapes
:€€€€€€€€€
0
_user_specified_nameconv2d_transpose_input
€

Ѓ
F__inference_sequential_3_layer_call_and_return_conditional_losses_8350

inputs
conv2d_3_8343
conv2d_3_8345
identityИҐ conv2d_3/StatefulPartitionedCallЦ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_8343conv2d_3_8345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_83092"
 conv2d_3/StatefulPartitionedCallТ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_82882!
max_pooling2d_3/PartitionedCallІ
IdentityIdentity(max_pooling2d_3/PartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
о
W
+__inference_concatenate_layer_call_fn_10202
inputs_0
inputs_1
identityЎ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
г
ё
F__inference_sequential_7_layer_call_and_return_conditional_losses_8823
conv2d_transpose_3_input
conv2d_transpose_3_8816
conv2d_transpose_3_8818
identityИҐ*conv2d_transpose_3/StatefulPartitionedCallм
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputconv2d_transpose_3_8816conv2d_transpose_3_8818*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87832,
*conv2d_transpose_3/StatefulPartitionedCallЃ
up_sampling2d_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_88062!
up_sampling2d_3/PartitionedCall√
IdentityIdentity(up_sampling2d_3/PartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€@@`
2
_user_specified_nameconv2d_transpose_3_input
ё
щ
D__inference_sequential_layer_call_and_return_conditional_losses_9912

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpЇ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¶
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
conv2d/ReluЅ
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolє
IdentityIdentitymax_pooling2d/MaxPool:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
£A
и
?__inference_model_layer_call_and_return_conditional_losses_9173
input_1
sequential_8943
sequential_8945
sequential_1_8966
sequential_1_8968
sequential_2_8989
sequential_2_8991
sequential_3_9012
sequential_3_9014
sequential_4_9035
sequential_4_9037
sequential_5_9074
sequential_5_9076
sequential_6_9111
sequential_6_9113
sequential_7_9148
sequential_7_9150
conv2d_transpose_4_9167
conv2d_transpose_4_9169
identityИҐ*conv2d_transpose_4/StatefulPartitionedCallҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallҐ$sequential_2/StatefulPartitionedCallҐ$sequential_3/StatefulPartitionedCallҐ$sequential_4/StatefulPartitionedCallҐ$sequential_5/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ$sequential_7/StatefulPartitionedCall°
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_8943sequential_8945*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80682$
"sequential/StatefulPartitionedCallѕ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_8966sequential_1_8968*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81622&
$sequential_1/StatefulPartitionedCall—
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_8989sequential_2_8991*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82562&
$sequential_2/StatefulPartitionedCall—
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_9012sequential_3_9014*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83502&
$sequential_3/StatefulPartitionedCallг
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_4_9035sequential_4_9037*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84742&
$sequential_4/StatefulPartitionedCallЇ
concatenate/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90472
concatenate/PartitionedCallЏ
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_5_9074sequential_5_9076*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_85982&
$sequential_5/StatefulPartitionedCallЊ
concatenate/PartitionedCall_1PartitionedCall-sequential_5/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90852
concatenate/PartitionedCall_1№
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_1:output:0sequential_6_9111sequential_6_9113*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87222&
$sequential_6/StatefulPartitionedCallЉ
concatenate/PartitionedCall_2PartitionedCall-sequential_6/StatefulPartitionedCall:output:0+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91222
concatenate/PartitionedCall_2№
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_2:output:0sequential_7_9148sequential_7_9150*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88462&
$sequential_7/StatefulPartitionedCallЪ
concatenate/PartitionedCall_3PartitionedCall-sequential_7/StatefulPartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ААA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91592
concatenate/PartitionedCall_3ъ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_3:output:0conv2d_transpose_4_9167conv2d_transpose_4_9169*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_89112,
*conv2d_transpose_4/StatefulPartitionedCallД
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0+^conv2d_transpose_4/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1
ч,
Є
G__inference_sequential_4_layer_call_and_return_conditional_losses_10132

inputs=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource
identityИҐ'conv2d_transpose/BiasAdd/ReadVariableOpҐ0conv2d_transpose/conv2d_transpose/ReadVariableOpf
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d_transpose/ShapeЦ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackЪ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1Ъ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2»
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/3ш
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackЪ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackЮ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1Ю
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2“
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1ж
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpҐ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transposeњ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp÷
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose/BiasAddУ
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_transpose/Relu}
up_sampling2d/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/ShapeР
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stackФ
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1Ф
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2Ґ
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/ConstЦ
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulБ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#conv2d_transpose/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighborф
IdentityIdentity;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
Г
F__inference_sequential_2_layer_call_and_return_conditional_losses_9996

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource
identityИҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOp∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOpЊ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2
conv2d_2/Relu«
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolњ
IdentityIdentity max_pooling2d_2/MaxPool:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
г
ё
F__inference_sequential_7_layer_call_and_return_conditional_losses_8833
conv2d_transpose_3_input
conv2d_transpose_3_8826
conv2d_transpose_3_8828
identityИҐ*conv2d_transpose_3/StatefulPartitionedCallм
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputconv2d_transpose_3_8826conv2d_transpose_3_8828*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87832,
*conv2d_transpose_3/StatefulPartitionedCallЃ
up_sampling2d_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_88062!
up_sampling2d_3/PartitionedCall√
IdentityIdentity(up_sampling2d_3/PartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€@@`
2
_user_specified_nameconv2d_transpose_3_input
џ
Т
+__inference_sequential_5_layer_call_fn_8605
conv2d_transpose_1_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_85982
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€
2
_user_specified_nameconv2d_transpose_1_input
О
o
E__inference_concatenate_layer_call_and_return_conditional_losses_9159

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЙ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:€€€€€€€€€ААA2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААA2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:€€€€€€€€€АА:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:YU
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ќ

№
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10499

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
’
Ж
1__inference_conv2d_transpose_2_layer_call_fn_8669

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86592
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
’
Ж
1__inference_conv2d_transpose_3_layer_call_fn_8793

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87832
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
©
J
.__inference_up_sampling2d_2_layer_call_fn_8688

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_86822
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ч
ґ
F__inference_sequential_3_layer_call_and_return_conditional_losses_8337
conv2d_3_input
conv2d_3_8330
conv2d_3_8332
identityИҐ conv2d_3/StatefulPartitionedCallЮ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_8330conv2d_3_8332*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_83092"
 conv2d_3/StatefulPartitionedCallТ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_82882!
max_pooling2d_3/PartitionedCallІ
IdentityIdentity(max_pooling2d_3/PartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_nameconv2d_3_input
Ж
e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8682

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ќ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul’
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
half_pixel_centers(2
resize/ResizeNearestNeighbor§
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Н
∆
F__inference_sequential_4_layer_call_and_return_conditional_losses_8474

inputs
conv2d_transpose_8467
conv2d_transpose_8469
identityИҐ(conv2d_transpose/StatefulPartitionedCall–
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_8467conv2d_transpose_8469*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84112*
(conv2d_transpose/StatefulPartitionedCall¶
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_84342
up_sampling2d/PartitionedCallњ
IdentityIdentity&up_sampling2d/PartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶
Б
,__inference_sequential_7_layer_call_fn_10448

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88652
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@`
 
_user_specified_nameinputs
љЄ
†
?__inference_model_layer_call_and_return_conditional_losses_9818

inputs4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource8
4sequential_1_conv2d_1_conv2d_readvariableop_resource9
5sequential_1_conv2d_1_biasadd_readvariableop_resource8
4sequential_2_conv2d_2_conv2d_readvariableop_resource9
5sequential_2_conv2d_2_biasadd_readvariableop_resource8
4sequential_3_conv2d_3_conv2d_readvariableop_resource9
5sequential_3_conv2d_3_biasadd_readvariableop_resourceJ
Fsequential_4_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_4_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_5_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_5_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_6_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_6_conv2d_transpose_2_biasadd_readvariableop_resourceL
Hsequential_7_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceC
?sequential_7_conv2d_transpose_3_biasadd_readvariableop_resource?
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_4_biasadd_readvariableop_resource
identityИҐ)conv2d_transpose_4/BiasAdd/ReadVariableOpҐ2conv2d_transpose_4/conv2d_transpose/ReadVariableOpҐ(sequential/conv2d/BiasAdd/ReadVariableOpҐ'sequential/conv2d/Conv2D/ReadVariableOpҐ,sequential_1/conv2d_1/BiasAdd/ReadVariableOpҐ+sequential_1/conv2d_1/Conv2D/ReadVariableOpҐ,sequential_2/conv2d_2/BiasAdd/ReadVariableOpҐ+sequential_2/conv2d_2/Conv2D/ReadVariableOpҐ,sequential_3/conv2d_3/BiasAdd/ReadVariableOpҐ+sequential_3/conv2d_3/Conv2D/ReadVariableOpҐ4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpҐ=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpҐ6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpҐ?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpҐ?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpҐ?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpЋ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpџ
sequential/conv2d/Conv2DConv2Dinputs/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2
sequential/conv2d/Conv2D¬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp“
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
sequential/conv2d/BiasAddШ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
sequential/conv2d/Reluв
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool„
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOpИ
sequential_1/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2Dќ
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpа
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
sequential_1/conv2d_1/BiasAddҐ
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
sequential_1/conv2d_1/Reluо
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_1/MaxPool„
+sequential_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_2/conv2d_2/Conv2D/ReadVariableOpМ
sequential_2/conv2d_2/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:03sequential_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2
sequential_2/conv2d_2/Conv2Dќ
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpа
sequential_2/conv2d_2/BiasAddBiasAdd%sequential_2/conv2d_2/Conv2D:output:04sequential_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
sequential_2/conv2d_2/BiasAddҐ
sequential_2/conv2d_2/ReluRelu&sequential_2/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2
sequential_2/conv2d_2/Reluо
$sequential_2/max_pooling2d_2/MaxPoolMaxPool(sequential_2/conv2d_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_2/MaxPool„
+sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_3/conv2d_3/Conv2D/ReadVariableOpМ
sequential_3/conv2d_3/Conv2DConv2D-sequential_2/max_pooling2d_2/MaxPool:output:03sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2
sequential_3/conv2d_3/Conv2Dќ
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpа
sequential_3/conv2d_3/BiasAddBiasAdd%sequential_3/conv2d_3/Conv2D:output:04sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential_3/conv2d_3/BiasAddҐ
sequential_3/conv2d_3/ReluRelu&sequential_3/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential_3/conv2d_3/Reluо
$sequential_3/max_pooling2d_3/MaxPoolMaxPool(sequential_3/conv2d_3/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_3/MaxPoolІ
#sequential_4/conv2d_transpose/ShapeShape-sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2%
#sequential_4/conv2d_transpose/Shape∞
1sequential_4/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_4/conv2d_transpose/strided_slice/stackі
3sequential_4/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_4/conv2d_transpose/strided_slice/stack_1і
3sequential_4/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_4/conv2d_transpose/strided_slice/stack_2Ц
+sequential_4/conv2d_transpose/strided_sliceStridedSlice,sequential_4/conv2d_transpose/Shape:output:0:sequential_4/conv2d_transpose/strided_slice/stack:output:0<sequential_4/conv2d_transpose/strided_slice/stack_1:output:0<sequential_4/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_4/conv2d_transpose/strided_sliceР
%sequential_4/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_4/conv2d_transpose/stack/1Р
%sequential_4/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_4/conv2d_transpose/stack/2Р
%sequential_4/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_4/conv2d_transpose/stack/3∆
#sequential_4/conv2d_transpose/stackPack4sequential_4/conv2d_transpose/strided_slice:output:0.sequential_4/conv2d_transpose/stack/1:output:0.sequential_4/conv2d_transpose/stack/2:output:0.sequential_4/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_4/conv2d_transpose/stackі
3sequential_4/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_4/conv2d_transpose/strided_slice_1/stackЄ
5sequential_4/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_4/conv2d_transpose/strided_slice_1/stack_1Є
5sequential_4/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_4/conv2d_transpose/strided_slice_1/stack_2†
-sequential_4/conv2d_transpose/strided_slice_1StridedSlice,sequential_4/conv2d_transpose/stack:output:0<sequential_4/conv2d_transpose/strided_slice_1/stack:output:0>sequential_4/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_4/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_4/conv2d_transpose/strided_slice_1Н
=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_4_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02?
=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOpэ
.sequential_4/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_4/conv2d_transpose/stack:output:0Esequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0-sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
20
.sequential_4/conv2d_transpose/conv2d_transposeж
4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_4_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOpК
%sequential_4/conv2d_transpose/BiasAddBiasAdd7sequential_4/conv2d_transpose/conv2d_transpose:output:0<sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%sequential_4/conv2d_transpose/BiasAddЇ
"sequential_4/conv2d_transpose/ReluRelu.sequential_4/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2$
"sequential_4/conv2d_transpose/Relu§
 sequential_4/up_sampling2d/ShapeShape0sequential_4/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 sequential_4/up_sampling2d/Shape™
.sequential_4/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.sequential_4/up_sampling2d/strided_slice/stackЃ
0sequential_4/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_4/up_sampling2d/strided_slice/stack_1Ѓ
0sequential_4/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_4/up_sampling2d/strided_slice/stack_2р
(sequential_4/up_sampling2d/strided_sliceStridedSlice)sequential_4/up_sampling2d/Shape:output:07sequential_4/up_sampling2d/strided_slice/stack:output:09sequential_4/up_sampling2d/strided_slice/stack_1:output:09sequential_4/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2*
(sequential_4/up_sampling2d/strided_sliceХ
 sequential_4/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2"
 sequential_4/up_sampling2d/Const 
sequential_4/up_sampling2d/mulMul1sequential_4/up_sampling2d/strided_slice:output:0)sequential_4/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2 
sequential_4/up_sampling2d/mulµ
7sequential_4/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor0sequential_4/conv2d_transpose/Relu:activations:0"sequential_4/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€*
half_pixel_centers(29
7sequential_4/up_sampling2d/resize/ResizeNearestNeighbort
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisТ
concatenate/concatConcatV2Hsequential_4/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0-sequential_2/max_pooling2d_2/MaxPool:output:0 concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€2
concatenate/concatЩ
%sequential_5/conv2d_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_1/Shapeі
3sequential_5/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_5/conv2d_transpose_1/strided_slice/stackЄ
5sequential_5/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_1/strided_slice/stack_1Є
5sequential_5/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/conv2d_transpose_1/strided_slice/stack_2Ґ
-sequential_5/conv2d_transpose_1/strided_sliceStridedSlice.sequential_5/conv2d_transpose_1/Shape:output:0<sequential_5/conv2d_transpose_1/strided_slice/stack:output:0>sequential_5/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_5/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/conv2d_transpose_1/strided_sliceФ
'sequential_5/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_1/stack/1Ф
'sequential_5/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_1/stack/2Ф
'sequential_5/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_5/conv2d_transpose_1/stack/3“
%sequential_5/conv2d_transpose_1/stackPack6sequential_5/conv2d_transpose_1/strided_slice:output:00sequential_5/conv2d_transpose_1/stack/1:output:00sequential_5/conv2d_transpose_1/stack/2:output:00sequential_5/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/conv2d_transpose_1/stackЄ
5sequential_5/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_5/conv2d_transpose_1/strided_slice_1/stackЉ
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_1Љ
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_5/conv2d_transpose_1/strided_slice_1/stack_2ђ
/sequential_5/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_5/conv2d_transpose_1/stack:output:0>sequential_5/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_5/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_5/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_5/conv2d_transpose_1/strided_slice_1У
?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_5_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02A
?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOpу
0sequential_5/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_5/conv2d_transpose_1/stack:output:0Gsequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
22
0sequential_5/conv2d_transpose_1/conv2d_transposeм
6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOpТ
'sequential_5/conv2d_transpose_1/BiasAddBiasAdd9sequential_5/conv2d_transpose_1/conv2d_transpose:output:0>sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'sequential_5/conv2d_transpose_1/BiasAddј
$sequential_5/conv2d_transpose_1/ReluRelu0sequential_5/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2&
$sequential_5/conv2d_transpose_1/Relu™
"sequential_5/up_sampling2d_1/ShapeShape2sequential_5/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential_5/up_sampling2d_1/ShapeЃ
0sequential_5/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_5/up_sampling2d_1/strided_slice/stack≤
2sequential_5/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_5/up_sampling2d_1/strided_slice/stack_1≤
2sequential_5/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_5/up_sampling2d_1/strided_slice/stack_2ь
*sequential_5/up_sampling2d_1/strided_sliceStridedSlice+sequential_5/up_sampling2d_1/Shape:output:09sequential_5/up_sampling2d_1/strided_slice/stack:output:0;sequential_5/up_sampling2d_1/strided_slice/stack_1:output:0;sequential_5/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_5/up_sampling2d_1/strided_sliceЩ
"sequential_5/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_5/up_sampling2d_1/Const“
 sequential_5/up_sampling2d_1/mulMul3sequential_5/up_sampling2d_1/strided_slice:output:0+sequential_5/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2"
 sequential_5/up_sampling2d_1/mulљ
9sequential_5/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_5/conv2d_transpose_1/Relu:activations:0$sequential_5/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€  *
half_pixel_centers(2;
9sequential_5/up_sampling2d_1/resize/ResizeNearestNeighborx
concatenate/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat_1/axisЪ
concatenate/concat_1ConcatV2Jsequential_5/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0-sequential_1/max_pooling2d_1/MaxPool:output:0"concatenate/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€  02
concatenate/concat_1Ы
%sequential_6/conv2d_transpose_2/ShapeShapeconcatenate/concat_1:output:0*
T0*
_output_shapes
:2'
%sequential_6/conv2d_transpose_2/Shapeі
3sequential_6/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_6/conv2d_transpose_2/strided_slice/stackЄ
5sequential_6/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_6/conv2d_transpose_2/strided_slice/stack_1Є
5sequential_6/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_6/conv2d_transpose_2/strided_slice/stack_2Ґ
-sequential_6/conv2d_transpose_2/strided_sliceStridedSlice.sequential_6/conv2d_transpose_2/Shape:output:0<sequential_6/conv2d_transpose_2/strided_slice/stack:output:0>sequential_6/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_6/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_6/conv2d_transpose_2/strided_sliceФ
'sequential_6/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/conv2d_transpose_2/stack/1Ф
'sequential_6/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/conv2d_transpose_2/stack/2Ф
'sequential_6/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/conv2d_transpose_2/stack/3“
%sequential_6/conv2d_transpose_2/stackPack6sequential_6/conv2d_transpose_2/strided_slice:output:00sequential_6/conv2d_transpose_2/stack/1:output:00sequential_6/conv2d_transpose_2/stack/2:output:00sequential_6/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/conv2d_transpose_2/stackЄ
5sequential_6/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_6/conv2d_transpose_2/strided_slice_1/stackЉ
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_1Љ
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_6/conv2d_transpose_2/strided_slice_1/stack_2ђ
/sequential_6/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_6/conv2d_transpose_2/stack:output:0>sequential_6/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_6/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_6/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_6/conv2d_transpose_2/strided_slice_1У
?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_6_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02A
?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOpх
0sequential_6/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_6/conv2d_transpose_2/stack:output:0Gsequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0concatenate/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
22
0sequential_6/conv2d_transpose_2/conv2d_transposeм
6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_6_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOpТ
'sequential_6/conv2d_transpose_2/BiasAddBiasAdd9sequential_6/conv2d_transpose_2/conv2d_transpose:output:0>sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2)
'sequential_6/conv2d_transpose_2/BiasAddј
$sequential_6/conv2d_transpose_2/ReluRelu0sequential_6/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2&
$sequential_6/conv2d_transpose_2/Relu™
"sequential_6/up_sampling2d_2/ShapeShape2sequential_6/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential_6/up_sampling2d_2/ShapeЃ
0sequential_6/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_6/up_sampling2d_2/strided_slice/stack≤
2sequential_6/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_6/up_sampling2d_2/strided_slice/stack_1≤
2sequential_6/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_6/up_sampling2d_2/strided_slice/stack_2ь
*sequential_6/up_sampling2d_2/strided_sliceStridedSlice+sequential_6/up_sampling2d_2/Shape:output:09sequential_6/up_sampling2d_2/strided_slice/stack:output:0;sequential_6/up_sampling2d_2/strided_slice/stack_1:output:0;sequential_6/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_6/up_sampling2d_2/strided_sliceЩ
"sequential_6/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_6/up_sampling2d_2/Const“
 sequential_6/up_sampling2d_2/mulMul3sequential_6/up_sampling2d_2/strided_slice:output:0+sequential_6/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2"
 sequential_6/up_sampling2d_2/mulљ
9sequential_6/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_6/conv2d_transpose_2/Relu:activations:0$sequential_6/up_sampling2d_2/mul:z:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
half_pixel_centers(2;
9sequential_6/up_sampling2d_2/resize/ResizeNearestNeighborx
concatenate/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat_2/axisЦ
concatenate/concat_2ConcatV2Jsequential_6/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0)sequential/max_pooling2d/MaxPool:output:0"concatenate/concat_2/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@`2
concatenate/concat_2Ы
%sequential_7/conv2d_transpose_3/ShapeShapeconcatenate/concat_2:output:0*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_3/Shapeі
3sequential_7/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_7/conv2d_transpose_3/strided_slice/stackЄ
5sequential_7/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_3/strided_slice/stack_1Є
5sequential_7/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_7/conv2d_transpose_3/strided_slice/stack_2Ґ
-sequential_7/conv2d_transpose_3/strided_sliceStridedSlice.sequential_7/conv2d_transpose_3/Shape:output:0<sequential_7/conv2d_transpose_3/strided_slice/stack:output:0>sequential_7/conv2d_transpose_3/strided_slice/stack_1:output:0>sequential_7/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_7/conv2d_transpose_3/strided_sliceФ
'sequential_7/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_3/stack/1Ф
'sequential_7/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_3/stack/2Ф
'sequential_7/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_7/conv2d_transpose_3/stack/3“
%sequential_7/conv2d_transpose_3/stackPack6sequential_7/conv2d_transpose_3/strided_slice:output:00sequential_7/conv2d_transpose_3/stack/1:output:00sequential_7/conv2d_transpose_3/stack/2:output:00sequential_7/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/conv2d_transpose_3/stackЄ
5sequential_7/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_7/conv2d_transpose_3/strided_slice_1/stackЉ
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_1Љ
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_7/conv2d_transpose_3/strided_slice_1/stack_2ђ
/sequential_7/conv2d_transpose_3/strided_slice_1StridedSlice.sequential_7/conv2d_transpose_3/stack:output:0>sequential_7/conv2d_transpose_3/strided_slice_1/stack:output:0@sequential_7/conv2d_transpose_3/strided_slice_1/stack_1:output:0@sequential_7/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_7/conv2d_transpose_3/strided_slice_1У
?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_7_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype02A
?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOpх
0sequential_7/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.sequential_7/conv2d_transpose_3/stack:output:0Gsequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0concatenate/concat_2:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@*
paddingSAME*
strides
22
0sequential_7/conv2d_transpose_3/conv2d_transposeм
6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_7_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOpТ
'sequential_7/conv2d_transpose_3/BiasAddBiasAdd9sequential_7/conv2d_transpose_3/conv2d_transpose:output:0>sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2)
'sequential_7/conv2d_transpose_3/BiasAddј
$sequential_7/conv2d_transpose_3/ReluRelu0sequential_7/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@@2&
$sequential_7/conv2d_transpose_3/Relu™
"sequential_7/up_sampling2d_3/ShapeShape2sequential_7/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential_7/up_sampling2d_3/ShapeЃ
0sequential_7/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_7/up_sampling2d_3/strided_slice/stack≤
2sequential_7/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_7/up_sampling2d_3/strided_slice/stack_1≤
2sequential_7/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_7/up_sampling2d_3/strided_slice/stack_2ь
*sequential_7/up_sampling2d_3/strided_sliceStridedSlice+sequential_7/up_sampling2d_3/Shape:output:09sequential_7/up_sampling2d_3/strided_slice/stack:output:0;sequential_7/up_sampling2d_3/strided_slice/stack_1:output:0;sequential_7/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*sequential_7/up_sampling2d_3/strided_sliceЩ
"sequential_7/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"sequential_7/up_sampling2d_3/Const“
 sequential_7/up_sampling2d_3/mulMul3sequential_7/up_sampling2d_3/strided_slice:output:0+sequential_7/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2"
 sequential_7/up_sampling2d_3/mulњ
9sequential_7/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor2sequential_7/conv2d_transpose_3/Relu:activations:0$sequential_7/up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
half_pixel_centers(2;
9sequential_7/up_sampling2d_3/resize/ResizeNearestNeighborx
concatenate/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat_3/axisх
concatenate/concat_3ConcatV2Jsequential_7/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0inputs"concatenate/concat_3/axis:output:0*
N*
T0*1
_output_shapes
:€€€€€€€€€ААA2
concatenate/concat_3Б
conv2d_transpose_4/ShapeShapeconcatenate/concat_3:output:0*
T0*
_output_shapes
:2
conv2d_transpose_4/ShapeЪ
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stackЮ
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1Ю
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2‘
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice{
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :А2
conv2d_transpose_4/stack/1{
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :А2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3Д
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stackЮ
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stackҐ
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1Ґ
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2ё
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1м
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:A*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpƒ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0concatenate/concat_3:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingVALID*
strides
2%
#conv2d_transpose_4/conv2d_transpose≈
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOpа
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_transpose_4/BiasAddЫ
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_transpose_4/ReluЄ
IdentityIdentity%conv2d_transpose_4/Relu:activations:0*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp-^sequential_2/conv2d_2/BiasAdd/ReadVariableOp,^sequential_2/conv2d_2/Conv2D/ReadVariableOp-^sequential_3/conv2d_3/BiasAdd/ReadVariableOp,^sequential_3/conv2d_3/Conv2D/ReadVariableOp5^sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp@^sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_2/BiasAdd/ReadVariableOp,sequential_2/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_2/Conv2D/ReadVariableOp+sequential_2/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_3/BiasAdd/ReadVariableOp,sequential_3/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_3/Conv2D/ReadVariableOp+sequential_3/conv2d_3/Conv2D/ReadVariableOp2l
4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_4/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_4/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_5/conv2d_transpose_1/BiasAdd/ReadVariableOp2В
?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_5/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_6/conv2d_transpose_2/BiasAdd/ReadVariableOp2В
?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_6/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp6sequential_7/conv2d_transpose_3/BiasAdd/ReadVariableOp2В
?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?sequential_7/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
€

Ѓ
F__inference_sequential_1_layer_call_and_return_conditional_losses_8181

inputs
conv2d_1_8174
conv2d_1_8176
identityИҐ conv2d_1/StatefulPartitionedCallЦ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_8174conv2d_1_8176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_81212"
 conv2d_1/StatefulPartitionedCallТ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_81002!
max_pooling2d_1/PartitionedCallІ
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
г

¶
D__inference_sequential_layer_call_and_return_conditional_losses_8068

inputs
conv2d_8061
conv2d_8063
identityИҐconv2d/StatefulPartitionedCallО
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8061conv2d_8063*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_80272 
conv2d/StatefulPartitionedCallК
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_80062
max_pooling2d/PartitionedCall£
IdentityIdentity&max_pooling2d/PartitionedCall:output:0^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
э
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8006

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Б
А
+__inference_sequential_1_layer_call_fn_9975

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81622
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
Ч
ґ
F__inference_sequential_3_layer_call_and_return_conditional_losses_8327
conv2d_3_input
conv2d_3_8320
conv2d_3_8322
identityИҐ conv2d_3/StatefulPartitionedCallЮ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_8320conv2d_3_8322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_83092"
 conv2d_3/StatefulPartitionedCallТ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_82882!
max_pooling2d_3/PartitionedCallІ
IdentityIdentity(max_pooling2d_3/PartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€
(
_user_specified_nameconv2d_3_input
В
Б
,__inference_sequential_3_layer_call_fn_10068

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83692
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶
Б
,__inference_sequential_5_layer_call_fn_10275

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_85982
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
В
Б
,__inference_sequential_2_layer_call_fn_10017

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
В
Б
,__inference_sequential_2_layer_call_fn_10026

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
Ч
r
F__inference_concatenate_layer_call_and_return_conditional_losses_10183
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:€€€€€€€€€ААA2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААA2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:€€€€€€€€€АА:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:€€€€€€€€€АА
"
_user_specified_name
inputs/1
щ
}
(__inference_conv2d_3_layer_call_fn_10528

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_83092
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
џ
Т
+__inference_sequential_6_layer_call_fn_8748
conv2d_transpose_2_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87412
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€  0
2
_user_specified_nameconv2d_transpose_2_input
х

ђ
D__inference_sequential_layer_call_and_return_conditional_losses_8055
conv2d_input
conv2d_8048
conv2d_8050
identityИҐconv2d/StatefulPartitionedCallФ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_8048conv2d_8050*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_80272 
conv2d/StatefulPartitionedCallК
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_80062
max_pooling2d/PartitionedCall£
IdentityIdentity&max_pooling2d/PartitionedCall:output:0^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:_ [
1
_output_shapes
:€€€€€€€€€АА
&
_user_specified_nameconv2d_input
ЯA
з
?__inference_model_layer_call_and_return_conditional_losses_9376

inputs
sequential_9326
sequential_9328
sequential_1_9331
sequential_1_9333
sequential_2_9336
sequential_2_9338
sequential_3_9341
sequential_3_9343
sequential_4_9346
sequential_4_9348
sequential_5_9352
sequential_5_9354
sequential_6_9358
sequential_6_9360
sequential_7_9364
sequential_7_9366
conv2d_transpose_4_9370
conv2d_transpose_4_9372
identityИҐ*conv2d_transpose_4/StatefulPartitionedCallҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallҐ$sequential_2/StatefulPartitionedCallҐ$sequential_3/StatefulPartitionedCallҐ$sequential_4/StatefulPartitionedCallҐ$sequential_5/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ$sequential_7/StatefulPartitionedCall†
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9326sequential_9328*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80872$
"sequential/StatefulPartitionedCallѕ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_9331sequential_1_9333*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81812&
$sequential_1/StatefulPartitionedCall—
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_9336sequential_2_9338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82752&
$sequential_2/StatefulPartitionedCall—
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_9341sequential_3_9343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83692&
$sequential_3/StatefulPartitionedCallг
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_4_9346sequential_4_9348*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84932&
$sequential_4/StatefulPartitionedCallЇ
concatenate/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90472
concatenate/PartitionedCallЏ
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_5_9352sequential_5_9354*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_86172&
$sequential_5/StatefulPartitionedCallЊ
concatenate/PartitionedCall_1PartitionedCall-sequential_5/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90852
concatenate/PartitionedCall_1№
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_1:output:0sequential_6_9358sequential_6_9360*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87412&
$sequential_6/StatefulPartitionedCallЉ
concatenate/PartitionedCall_2PartitionedCall-sequential_6/StatefulPartitionedCall:output:0+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91222
concatenate/PartitionedCall_2№
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_2:output:0sequential_7_9364sequential_7_9366*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88652&
$sequential_7/StatefulPartitionedCallЩ
concatenate/PartitionedCall_3PartitionedCall-sequential_7/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ААA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91592
concatenate/PartitionedCall_3ъ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_3:output:0conv2d_transpose_4_9370conv2d_transpose_4_9372*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_89112,
*conv2d_transpose_4/StatefulPartitionedCallД
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0+^conv2d_transpose_4/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ж
o
E__inference_concatenate_layer_call_and_return_conditional_losses_9047

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЗ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
У
Д
G__inference_sequential_2_layer_call_and_return_conditional_losses_10008

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource
identityИҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOp∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOpЊ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2
conv2d_2/Relu«
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolњ
IdentityIdentity max_pooling2d_2/MaxPool:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
г
ё
F__inference_sequential_6_layer_call_and_return_conditional_losses_8699
conv2d_transpose_2_input
conv2d_transpose_2_8692
conv2d_transpose_2_8694
identityИҐ*conv2d_transpose_2/StatefulPartitionedCallм
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputconv2d_transpose_2_8692conv2d_transpose_2_8694*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86592,
*conv2d_transpose_2/StatefulPartitionedCallЃ
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_86822!
up_sampling2d_2/PartitionedCall√
IdentityIdentity(up_sampling2d_2/PartitionedCall:output:0+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€  0
2
_user_specified_nameconv2d_transpose_2_input
©
J
.__inference_up_sampling2d_3_layer_call_fn_8812

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_88062
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
Д
)__inference_sequential_layer_call_fn_8075
conv2d_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80682
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:€€€€€€€€€АА
&
_user_specified_nameconv2d_input
©
J
.__inference_max_pooling2d_3_layer_call_fn_8294

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_82882
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
Д
)__inference_sequential_layer_call_fn_8094
conv2d_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:€€€€€€€€€АА
&
_user_specified_nameconv2d_input
џ
Т
+__inference_sequential_5_layer_call_fn_8624
conv2d_transpose_1_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_86172
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€
2
_user_specified_nameconv2d_transpose_1_input
Т
Г
F__inference_sequential_1_layer_call_and_return_conditional_losses_9966

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOp∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_1/Conv2D/ReadVariableOpЊ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2d_1/Relu«
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolњ
IdentityIdentity max_pooling2d_1/MaxPool:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
ЇЕ
и
__inference__traced_save_10749
file_prefix8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_total_confusion_matrix_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameљ#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*ѕ"
value≈"B¬"CB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBEkeras_api/metrics/3/total_confusion_matrix/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesС
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesс
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_total_confusion_matrix_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ѕ
_input_shapesѓ
ђ: :A:: : : : : :@:@:@ : : :::::::: 0: :@`:@: : : : : : ::A::@:@:@ : : :::::::: 0: :@`:@:A::@:@:@ : : :::::::: 0: :@`:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:A: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: 0: 

_output_shapes
: :,(
&
_output_shapes
:@`: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
:A:  

_output_shapes
::,!(
&
_output_shapes
:@: "

_output_shapes
:@:,#(
&
_output_shapes
:@ : $

_output_shapes
: :,%(
&
_output_shapes
: : &

_output_shapes
::,'(
&
_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
::,+(
&
_output_shapes
:: ,

_output_shapes
::,-(
&
_output_shapes
: 0: .

_output_shapes
: :,/(
&
_output_shapes
:@`: 0

_output_shapes
:@:,1(
&
_output_shapes
:A: 2

_output_shapes
::,3(
&
_output_shapes
:@: 4

_output_shapes
:@:,5(
&
_output_shapes
:@ : 6

_output_shapes
: :,7(
&
_output_shapes
: : 8

_output_shapes
::,9(
&
_output_shapes
:: :

_output_shapes
::,;(
&
_output_shapes
:: <

_output_shapes
::,=(
&
_output_shapes
:: >

_output_shapes
::,?(
&
_output_shapes
: 0: @

_output_shapes
: :,A(
&
_output_shapes
:@`: B

_output_shapes
:@:C

_output_shapes
: 
Ч
ґ
F__inference_sequential_2_layer_call_and_return_conditional_losses_8243
conv2d_2_input
conv2d_2_8236
conv2d_2_8238
identityИҐ conv2d_2/StatefulPartitionedCallЮ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_8236conv2d_2_8238*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_82152"
 conv2d_2/StatefulPartitionedCallТ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_81942!
max_pooling2d_2/PartitionedCallІ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€   
(
_user_specified_nameconv2d_2_input
€

Ѓ
F__inference_sequential_2_layer_call_and_return_conditional_losses_8256

inputs
conv2d_2_8249
conv2d_2_8251
identityИҐ conv2d_2/StatefulPartitionedCallЦ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_8249conv2d_2_8251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_82152"
 conv2d_2/StatefulPartitionedCallТ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_81942!
max_pooling2d_2/PartitionedCallІ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
¶
Б
,__inference_sequential_7_layer_call_fn_10439

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88462
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@`
 
_user_specified_nameinputs
э
{
&__inference_conv2d_layer_call_fn_10468

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_80272
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
о
W
+__inference_concatenate_layer_call_fn_10163
inputs_0
inputs_1
identityЎ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91222
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@`2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :€€€€€€€€€@@@:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€@@@
"
_user_specified_name
inputs/1
ё
щ
D__inference_sequential_layer_call_and_return_conditional_losses_9924

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpЇ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@*
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¶
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА@2
conv2d/ReluЅ
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolє
IdentityIdentitymax_pooling2d/MaxPool:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Щ
И
+__inference_sequential_1_layer_call_fn_8188
conv2d_1_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81812
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€@@@
(
_user_specified_nameconv2d_1_input
≠
ћ
F__inference_sequential_7_layer_call_and_return_conditional_losses_8865

inputs
conv2d_transpose_3_8858
conv2d_transpose_3_8860
identityИҐ*conv2d_transpose_3/StatefulPartitionedCallЏ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_8858conv2d_transpose_3_8860*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87832,
*conv2d_transpose_3/StatefulPartitionedCallЃ
up_sampling2d_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_88062!
up_sampling2d_3/PartitionedCall√
IdentityIdentity(up_sampling2d_3/PartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@`
 
_user_specified_nameinputs
Н
∆
F__inference_sequential_4_layer_call_and_return_conditional_losses_8493

inputs
conv2d_transpose_8486
conv2d_transpose_8488
identityИҐ(conv2d_transpose/StatefulPartitionedCall–
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_8486conv2d_transpose_8488*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84112*
(conv2d_transpose/StatefulPartitionedCall¶
up_sampling2d/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_84342
up_sampling2d/PartitionedCallњ
IdentityIdentity&up_sampling2d/PartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Щ
И
+__inference_sequential_2_layer_call_fn_8263
conv2d_2_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€   
(
_user_specified_nameconv2d_2_input
ћ

џ
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8215

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
Ж
o
E__inference_concatenate_layer_call_and_return_conditional_losses_9122

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЗ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@`2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@`2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :€€€€€€€€€@@@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
а$
щ
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_8659

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
В
Б
,__inference_sequential_3_layer_call_fn_10059

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83502
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
ћ
F__inference_sequential_5_layer_call_and_return_conditional_losses_8598

inputs
conv2d_transpose_1_8591
conv2d_transpose_1_8593
identityИҐ*conv2d_transpose_1/StatefulPartitionedCallЏ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_1_8591conv2d_transpose_1_8593*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_85352,
*conv2d_transpose_1/StatefulPartitionedCallЃ
up_sampling2d_1/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_85582!
up_sampling2d_1/PartitionedCall√
IdentityIdentity(up_sampling2d_1/PartitionedCall:output:0+^conv2d_transpose_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
—
Д
/__inference_conv2d_transpose_layer_call_fn_8421

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84112
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
©
р
$__inference_model_layer_call_fn_9900

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_93762
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
г
ё
F__inference_sequential_6_layer_call_and_return_conditional_losses_8709
conv2d_transpose_2_input
conv2d_transpose_2_8702
conv2d_transpose_2_8704
identityИҐ*conv2d_transpose_2/StatefulPartitionedCallм
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputconv2d_transpose_2_8702conv2d_transpose_2_8704*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86592,
*conv2d_transpose_2/StatefulPartitionedCallЃ
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_86822!
up_sampling2d_2/PartitionedCall√
IdentityIdentity(up_sampling2d_2/PartitionedCall:output:0+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€  0
2
_user_specified_nameconv2d_transpose_2_input
≠
ћ
F__inference_sequential_7_layer_call_and_return_conditional_losses_8846

inputs
conv2d_transpose_3_8839
conv2d_transpose_3_8841
identityИҐ*conv2d_transpose_3/StatefulPartitionedCallЏ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_8839conv2d_transpose_3_8841*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_87832,
*conv2d_transpose_3/StatefulPartitionedCallЃ
up_sampling2d_3/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_88062!
up_sampling2d_3/PartitionedCall√
IdentityIdentity(up_sampling2d_3/PartitionedCall:output:0+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@`
 
_user_specified_nameinputs
Т
Г
F__inference_sequential_1_layer_call_and_return_conditional_losses_9954

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOp∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_1/Conv2D/ReadVariableOpЊ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2d_1/Relu«
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolњ
IdentityIdentity max_pooling2d_1/MaxPool:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
’
Р
+__inference_sequential_4_layer_call_fn_8481
conv2d_transpose_input
unknown
	unknown_0
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84742
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:g c
/
_output_shapes
:€€€€€€€€€
0
_user_specified_nameconv2d_transpose_input
•
H
,__inference_max_pooling2d_layer_call_fn_8012

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_80062
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
©
J
.__inference_up_sampling2d_1_layer_call_fn_8564

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_85582
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ

џ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8121

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
€

Ѓ
F__inference_sequential_1_layer_call_and_return_conditional_losses_8162

inputs
conv2d_1_8155
conv2d_1_8157
identityИҐ conv2d_1/StatefulPartitionedCallЦ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_8155conv2d_1_8157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_81212"
 conv2d_1/StatefulPartitionedCallТ
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_81002!
max_pooling2d_1/PartitionedCallІ
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@@::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@@
 
_user_specified_nameinputs
а$
щ
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8535

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
к

п
"__inference_signature_wrapper_9466
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_80002
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1
€
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8100

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ЯA
з
?__inference_model_layer_call_and_return_conditional_losses_9282

inputs
sequential_9232
sequential_9234
sequential_1_9237
sequential_1_9239
sequential_2_9242
sequential_2_9244
sequential_3_9247
sequential_3_9249
sequential_4_9252
sequential_4_9254
sequential_5_9258
sequential_5_9260
sequential_6_9264
sequential_6_9266
sequential_7_9270
sequential_7_9272
conv2d_transpose_4_9276
conv2d_transpose_4_9278
identityИҐ*conv2d_transpose_4/StatefulPartitionedCallҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallҐ$sequential_2/StatefulPartitionedCallҐ$sequential_3/StatefulPartitionedCallҐ$sequential_4/StatefulPartitionedCallҐ$sequential_5/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ$sequential_7/StatefulPartitionedCall†
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9232sequential_9234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80682$
"sequential/StatefulPartitionedCallѕ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_9237sequential_1_9239*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81622&
$sequential_1/StatefulPartitionedCall—
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_9242sequential_2_9244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82562&
$sequential_2/StatefulPartitionedCall—
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_9247sequential_3_9249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83502&
$sequential_3/StatefulPartitionedCallг
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_4_9252sequential_4_9254*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84742&
$sequential_4/StatefulPartitionedCallЇ
concatenate/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90472
concatenate/PartitionedCallЏ
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_5_9258sequential_5_9260*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_85982&
$sequential_5/StatefulPartitionedCallЊ
concatenate/PartitionedCall_1PartitionedCall-sequential_5/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90852
concatenate/PartitionedCall_1№
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_1:output:0sequential_6_9264sequential_6_9266*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87222&
$sequential_6/StatefulPartitionedCallЉ
concatenate/PartitionedCall_2PartitionedCall-sequential_6/StatefulPartitionedCall:output:0+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91222
concatenate/PartitionedCall_2№
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_2:output:0sequential_7_9270sequential_7_9272*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88462&
$sequential_7/StatefulPartitionedCallЩ
concatenate/PartitionedCall_3PartitionedCall-sequential_7/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ААA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91592
concatenate/PartitionedCall_3ъ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_3:output:0conv2d_transpose_4_9276conv2d_transpose_4_9278*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_89112,
*conv2d_transpose_4/StatefulPartitionedCallД
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0+^conv2d_transpose_4/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
≠
ћ
F__inference_sequential_6_layer_call_and_return_conditional_losses_8722

inputs
conv2d_transpose_2_8715
conv2d_transpose_2_8717
identityИҐ*conv2d_transpose_2/StatefulPartitionedCallЏ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_8715conv2d_transpose_2_8717*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86592,
*conv2d_transpose_2/StatefulPartitionedCallЃ
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_86822!
up_sampling2d_2/PartitionedCall√
IdentityIdentity(up_sampling2d_2/PartitionedCall:output:0+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  0
 
_user_specified_nameinputs
≠
ћ
F__inference_sequential_6_layer_call_and_return_conditional_losses_8741

inputs
conv2d_transpose_2_8734
conv2d_transpose_2_8736
identityИҐ*conv2d_transpose_2/StatefulPartitionedCallЏ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_8734conv2d_transpose_2_8736*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_86592,
*conv2d_transpose_2/StatefulPartitionedCallЃ
up_sampling2d_2/PartitionedCallPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_86822!
up_sampling2d_2/PartitionedCall√
IdentityIdentity(up_sampling2d_2/PartitionedCall:output:0+^conv2d_transpose_2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  0
 
_user_specified_nameinputs
х

ђ
D__inference_sequential_layer_call_and_return_conditional_losses_8045
conv2d_input
conv2d_8038
conv2d_8040
identityИҐconv2d/StatefulPartitionedCallФ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_8038conv2d_8040*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_80272 
conv2d/StatefulPartitionedCallК
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_80062
max_pooling2d/PartitionedCall£
IdentityIdentity&max_pooling2d/PartitionedCall:output:0^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:_ [
1
_output_shapes
:€€€€€€€€€АА
&
_user_specified_nameconv2d_input
¶
Б
,__inference_sequential_6_layer_call_fn_10357

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87222
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€  0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  0
 
_user_specified_nameinputs
П
r
F__inference_concatenate_layer_call_and_return_conditional_losses_10196
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЙ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
€
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8288

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ё$
ч
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8411

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
џ
Т
+__inference_sequential_7_layer_call_fn_8872
conv2d_transpose_3_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88652
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@`::22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:€€€€€€€€€@@`
2
_user_specified_nameconv2d_transpose_3_input
£A
и
?__inference_model_layer_call_and_return_conditional_losses_9226
input_1
sequential_9176
sequential_9178
sequential_1_9181
sequential_1_9183
sequential_2_9186
sequential_2_9188
sequential_3_9191
sequential_3_9193
sequential_4_9196
sequential_4_9198
sequential_5_9202
sequential_5_9204
sequential_6_9208
sequential_6_9210
sequential_7_9214
sequential_7_9216
conv2d_transpose_4_9220
conv2d_transpose_4_9222
identityИҐ*conv2d_transpose_4/StatefulPartitionedCallҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallҐ$sequential_2/StatefulPartitionedCallҐ$sequential_3/StatefulPartitionedCallҐ$sequential_4/StatefulPartitionedCallҐ$sequential_5/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ$sequential_7/StatefulPartitionedCall°
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_9176sequential_9178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_80872$
"sequential/StatefulPartitionedCallѕ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_9181sequential_1_9183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_81812&
$sequential_1/StatefulPartitionedCall—
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_9186sequential_2_9188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_82752&
$sequential_2/StatefulPartitionedCall—
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_9191sequential_3_9193*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_3_layer_call_and_return_conditional_losses_83692&
$sequential_3/StatefulPartitionedCallг
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall-sequential_3/StatefulPartitionedCall:output:0sequential_4_9196sequential_4_9198*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84932&
$sequential_4/StatefulPartitionedCallЇ
concatenate/PartitionedCallPartitionedCall-sequential_4/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90472
concatenate/PartitionedCallЏ
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_5_9202sequential_5_9204*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_86172&
$sequential_5/StatefulPartitionedCallЊ
concatenate/PartitionedCall_1PartitionedCall-sequential_5/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_90852
concatenate/PartitionedCall_1№
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_1:output:0sequential_6_9208sequential_6_9210*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_87412&
$sequential_6/StatefulPartitionedCallЉ
concatenate/PartitionedCall_2PartitionedCall-sequential_6/StatefulPartitionedCall:output:0+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91222
concatenate/PartitionedCall_2№
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_2:output:0sequential_7_9214sequential_7_9216*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_88652&
$sequential_7/StatefulPartitionedCallЪ
concatenate/PartitionedCall_3PartitionedCall-sequential_7/StatefulPartitionedCall:output:0input_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ААA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_91592
concatenate/PartitionedCall_3ъ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate/PartitionedCall_3:output:0conv2d_transpose_4_9220conv2d_transpose_4_9222*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_89112,
*conv2d_transpose_4/StatefulPartitionedCallД
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0+^conv2d_transpose_4/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1
¶
Б
,__inference_sequential_4_layer_call_fn_10150

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_4_layer_call_and_return_conditional_losses_84932
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
У
Д
G__inference_sequential_3_layer_call_and_return_conditional_losses_10050

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identityИҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOp∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpЊ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_3/Relu«
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolњ
IdentityIdentity max_pooling2d_3/MaxPool:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а$
щ
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8783

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@`*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
У
Д
G__inference_sequential_3_layer_call_and_return_conditional_losses_10038

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identityИҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOp∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpЊ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_3/Relu«
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolњ
IdentityIdentity max_pooling2d_3/MaxPool:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*…
serving_defaultµ
E
input_1:
serving_default_input_1:0€€€€€€€€€ААP
conv2d_transpose_4:
StatefulPartitionedCall:0€€€€€€€€€ААtensorflow/serving/predict:ъЗ
у»
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+б&call_and_return_all_conditional_losses
в_default_save_signature
г__call__"™ƒ
_tf_keras_networkНƒ{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_1", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_2", "inbound_nodes": [[["sequential_1", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_3", "inbound_nodes": [[["sequential_2", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_4", "inbound_nodes": [[["sequential_3", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate", "inbound_nodes": [[["sequential_4", 1, 0, {}], ["sequential_2", 1, 0, {}]], [["sequential_5", 1, 0, {}], ["sequential_1", 1, 0, {}]], [["sequential_6", 1, 0, {}], ["sequential", 1, 0, {}]], [["sequential_7", 1, 0, {}], ["input_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_1_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_5", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 48]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_2_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_6", "inbound_nodes": [[["concatenate", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_3_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_7", "inbound_nodes": [[["concatenate", 2, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["concatenate", 3, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_transpose_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_1", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_2", "inbound_nodes": [[["sequential_1", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_3", "inbound_nodes": [[["sequential_2", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_4", "inbound_nodes": [[["sequential_3", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate", "inbound_nodes": [[["sequential_4", 1, 0, {}], ["sequential_2", 1, 0, {}]], [["sequential_5", 1, 0, {}], ["sequential_1", 1, 0, {}]], [["sequential_6", 1, 0, {}], ["sequential", 1, 0, {}]], [["sequential_7", 1, 0, {}], ["input_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_1_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_5", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 48]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_2_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_6", "inbound_nodes": [[["concatenate", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_3_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "name": "sequential_7", "inbound_nodes": [[["concatenate", 2, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["concatenate", 3, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_transpose_4", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [[{"class_name": "Accuracy", "config": {"name": "acc", "dtype": "float32"}}, {"class_name": "MeanSquaredError", "config": {"name": "mse", "dtype": "float32"}}, {"class_name": "MeanIoU", "config": {"name": "iou", "dtype": "float32", "num_classes": 4}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
э"ъ
_tf_keras_input_layerЏ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ѕ
layer_with_weights-0
layer-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
+д&call_and_return_all_conditional_losses
е__call__"К
_tf_keras_sequentialл{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
я
layer_with_weights-0
layer-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"Ъ
_tf_keras_sequentialы{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
я
layer_with_weights-0
layer-0
layer-1
 trainable_variables
!regularization_losses
"	variables
#	keras_api
+и&call_and_return_all_conditional_losses
й__call__"Ъ
_tf_keras_sequentialы{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
Ё
$layer_with_weights-0
$layer-0
%layer-1
&trainable_variables
'regularization_losses
(	variables
)	keras_api
+к&call_and_return_all_conditional_losses
л__call__"Ш
_tf_keras_sequentialщ{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
≈
*layer_with_weights-0
*layer-0
+layer-1
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+м&call_and_return_all_conditional_losses
н__call__"А
_tf_keras_sequentialб{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}}}
ў
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+о&call_and_return_all_conditional_losses
п__call__"»
_tf_keras_layerЃ{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 16, 8]}, {"class_name": "TensorShape", "items": [null, 16, 16, 16]}]}
Ё
4layer_with_weights-0
4layer-0
5layer-1
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+р&call_and_return_all_conditional_losses
с__call__"Ш
_tf_keras_sequentialщ{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_1_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 24]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_1_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}}}
Ё
:layer_with_weights-0
:layer-0
;layer-1
<trainable_variables
=regularization_losses
>	variables
?	keras_api
+т&call_and_return_all_conditional_losses
у__call__"Ш
_tf_keras_sequentialщ{"class_name": "Sequential", "name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 48]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_2_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 48]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 48]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_2_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}}}
Ё
@layer_with_weights-0
@layer-0
Alayer-1
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"Ш
_tf_keras_sequentialщ{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_3_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 96]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_transpose_3_input"}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}}]}}}
Ђ


Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"Д	
_tf_keras_layerк{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 65}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 65]}}
ї
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_rateFmљGmЊQmњRmјSmЅTm¬Um√VmƒWm≈Xm∆Ym«Zm»[m…\m ]mЋ^mћ_mЌ`mќFvѕGv–Qv—Rv“Sv”Tv‘Uv’Vv÷Wv„XvЎYvўZvЏ[vџ\v№]vЁ^vё_vя`vа"
	optimizer
¶
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15
F16
G17"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
Q0
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12
^13
_14
`15
F16
G17"
trackable_list_wrapper
ќ
alayer_regularization_losses
blayer_metrics
cnon_trainable_variables
trainable_variables

dlayers
regularization_losses
emetrics
	variables
г__call__
в_default_save_signature
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
-
шserving_default"
signature_map
р	

Qkernel
Rbias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"…
_tf_keras_layerѓ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
э
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"м
_tf_keras_layer“{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
∞
nlayer_regularization_losses
olayer_metrics
pnon_trainable_variables
trainable_variables

qlayers
regularization_losses
rmetrics
	variables
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
ф	

Skernel
Tbias
strainable_variables
tregularization_losses
u	variables
v	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"Ќ
_tf_keras_layer≥{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
Б
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+€&call_and_return_all_conditional_losses
А__call__"р
_tf_keras_layer÷{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
∞
{layer_regularization_losses
|layer_metrics
}non_trainable_variables
trainable_variables

~layers
regularization_losses
metrics
	variables
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
ш	

Ukernel
Vbias
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Ќ
_tf_keras_layer≥{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
Е
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"р
_tf_keras_layer÷{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
µ
 Иlayer_regularization_losses
Йlayer_metrics
Кnon_trainable_variables
 trainable_variables
Лlayers
!regularization_losses
Мmetrics
"	variables
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
ч	

Wkernel
Xbias
Нtrainable_variables
Оregularization_losses
П	variables
Р	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"ћ
_tf_keras_layer≤{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
Е
Сtrainable_variables
Тregularization_losses
У	variables
Ф	keras_api
+З&call_and_return_all_conditional_losses
И__call__"р
_tf_keras_layer÷{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
µ
 Хlayer_regularization_losses
Цlayer_metrics
Чnon_trainable_variables
&trainable_variables
Шlayers
'regularization_losses
Щmetrics
(	variables
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
§


Ykernel
Zbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"щ
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 8]}}
Ћ
Юtrainable_variables
Яregularization_losses
†	variables
°	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"ґ
_tf_keras_layerЬ{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
µ
 Ґlayer_regularization_losses
£layer_metrics
§non_trainable_variables
,trainable_variables
•layers
-regularization_losses
¶metrics
.	variables
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Іlayer_regularization_losses
®layer_metrics
©non_trainable_variables
0trainable_variables
™layers
1regularization_losses
Ђmetrics
2	variables
п__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
≠


[kernel
\bias
ђtrainable_variables
≠regularization_losses
Ѓ	variables
ѓ	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"В	
_tf_keras_layerи{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 24]}}
ѕ
∞trainable_variables
±regularization_losses
≤	variables
≥	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Ї
_tf_keras_layer†{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
µ
 іlayer_regularization_losses
µlayer_metrics
ґnon_trainable_variables
6trainable_variables
Јlayers
7regularization_losses
Єmetrics
8	variables
с__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
≠


]kernel
^bias
єtrainable_variables
Їregularization_losses
ї	variables
Љ	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"В	
_tf_keras_layerи{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 48]}}
ѕ
љtrainable_variables
Њregularization_losses
њ	variables
ј	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Ї
_tf_keras_layer†{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
 Ѕlayer_regularization_losses
¬layer_metrics
√non_trainable_variables
<trainable_variables
ƒlayers
=regularization_losses
≈metrics
>	variables
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
≠


_kernel
`bias
∆trainable_variables
«regularization_losses
»	variables
…	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"В	
_tf_keras_layerи{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 96]}}
ѕ
 trainable_variables
Ћregularization_losses
ћ	variables
Ќ	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"Ї
_tf_keras_layer†{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
µ
 ќlayer_regularization_losses
ѕlayer_metrics
–non_trainable_variables
Btrainable_variables
—layers
Cregularization_losses
“metrics
D	variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
3:1A2conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
µ
 ”layer_regularization_losses
‘layer_metrics
’non_trainable_variables
Htrainable_variables
÷layers
Iregularization_losses
„metrics
J	variables
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%@2conv2d/kernel
:@2conv2d/bias
):'@ 2conv2d_1/kernel
: 2conv2d_1/bias
):' 2conv2d_2/kernel
:2conv2d_2/bias
):'2conv2d_3/kernel
:2conv2d_3/bias
1:/2conv2d_transpose/kernel
#:!2conv2d_transpose/bias
3:12conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
3:1 02conv2d_transpose_2/kernel
%:# 2conv2d_transpose_2/bias
3:1@`2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
@
Ў0
ў1
Џ2
џ3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
µ
 №layer_regularization_losses
Ёlayer_metrics
ёnon_trainable_variables
ftrainable_variables
яlayers
gregularization_losses
аmetrics
h	variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 бlayer_regularization_losses
вlayer_metrics
гnon_trainable_variables
jtrainable_variables
дlayers
kregularization_losses
еmetrics
l	variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
µ
 жlayer_regularization_losses
зlayer_metrics
иnon_trainable_variables
strainable_variables
йlayers
tregularization_losses
кmetrics
u	variables
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 лlayer_regularization_losses
мlayer_metrics
нnon_trainable_variables
wtrainable_variables
оlayers
xregularization_losses
пmetrics
y	variables
А__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
Є
 рlayer_regularization_losses
сlayer_metrics
тnon_trainable_variables
Аtrainable_variables
уlayers
Бregularization_losses
фmetrics
В	variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 хlayer_regularization_losses
цlayer_metrics
чnon_trainable_variables
Дtrainable_variables
шlayers
Еregularization_losses
щmetrics
Ж	variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
Є
 ъlayer_regularization_losses
ыlayer_metrics
ьnon_trainable_variables
Нtrainable_variables
эlayers
Оregularization_losses
юmetrics
П	variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 €layer_regularization_losses
Аlayer_metrics
Бnon_trainable_variables
Сtrainable_variables
Вlayers
Тregularization_losses
Гmetrics
У	variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
Є
 Дlayer_regularization_losses
Еlayer_metrics
Жnon_trainable_variables
Ъtrainable_variables
Зlayers
Ыregularization_losses
Иmetrics
Ь	variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Йlayer_regularization_losses
Кlayer_metrics
Лnon_trainable_variables
Юtrainable_variables
Мlayers
Яregularization_losses
Нmetrics
†	variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
Є
 Оlayer_regularization_losses
Пlayer_metrics
Рnon_trainable_variables
ђtrainable_variables
Сlayers
≠regularization_losses
Тmetrics
Ѓ	variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Уlayer_regularization_losses
Фlayer_metrics
Хnon_trainable_variables
∞trainable_variables
Цlayers
±regularization_losses
Чmetrics
≤	variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
Є
 Шlayer_regularization_losses
Щlayer_metrics
Ъnon_trainable_variables
єtrainable_variables
Ыlayers
Їregularization_losses
Ьmetrics
ї	variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Эlayer_regularization_losses
Юlayer_metrics
Яnon_trainable_variables
љtrainable_variables
†layers
Њregularization_losses
°metrics
њ	variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
Є
 Ґlayer_regularization_losses
£layer_metrics
§non_trainable_variables
∆trainable_variables
•layers
«regularization_losses
¶metrics
»	variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Іlayer_regularization_losses
®layer_metrics
©non_trainable_variables
 trainable_variables
™layers
Ћregularization_losses
Ђmetrics
ћ	variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
њ

ђtotal

≠count
Ѓ	variables
ѓ	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
“

∞total

±count
≤
_fn_kwargs
≥	variables
і	keras_api"Ж
_tf_keras_metricl{"class_name": "Accuracy", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32"}}
Џ

µtotal

ґcount
Ј
_fn_kwargs
Є	variables
є	keras_api"О
_tf_keras_metrict{"class_name": "MeanSquaredError", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32"}}
ж
Їtotal_confusion_matrix
Їtotal_cm
ї	variables
Љ	keras_api"Ч
_tf_keras_metric}{"class_name": "MeanIoU", "name": "iou", "dtype": "float32", "config": {"name": "iou", "dtype": "float32", "num_classes": 4}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
ђ0
≠1"
trackable_list_wrapper
.
Ѓ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
∞0
±1"
trackable_list_wrapper
.
≥	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
µ0
ґ1"
trackable_list_wrapper
.
Є	variables"
_generic_user_object
*:( (2total_confusion_matrix
(
Ї0"
trackable_list_wrapper
.
ї	variables"
_generic_user_object
8:6A2 Adam/conv2d_transpose_4/kernel/m
*:(2Adam/conv2d_transpose_4/bias/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
.:,@ 2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, 2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
6:42Adam/conv2d_transpose/kernel/m
(:&2Adam/conv2d_transpose/bias/m
8:62 Adam/conv2d_transpose_1/kernel/m
*:(2Adam/conv2d_transpose_1/bias/m
8:6 02 Adam/conv2d_transpose_2/kernel/m
*:( 2Adam/conv2d_transpose_2/bias/m
8:6@`2 Adam/conv2d_transpose_3/kernel/m
*:(@2Adam/conv2d_transpose_3/bias/m
8:6A2 Adam/conv2d_transpose_4/kernel/v
*:(2Adam/conv2d_transpose_4/bias/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
.:,@ 2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, 2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
6:42Adam/conv2d_transpose/kernel/v
(:&2Adam/conv2d_transpose/bias/v
8:62 Adam/conv2d_transpose_1/kernel/v
*:(2Adam/conv2d_transpose_1/bias/v
8:6 02 Adam/conv2d_transpose_2/kernel/v
*:( 2Adam/conv2d_transpose_2/bias/v
8:6@`2 Adam/conv2d_transpose_3/kernel/v
*:(@2Adam/conv2d_transpose_3/bias/v
 2«
?__inference_model_layer_call_and_return_conditional_losses_9818
?__inference_model_layer_call_and_return_conditional_losses_9642
?__inference_model_layer_call_and_return_conditional_losses_9173
?__inference_model_layer_call_and_return_conditional_losses_9226ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
__inference__wrapped_model_8000ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *0Ґ-
+К(
input_1€€€€€€€€€АА
ё2џ
$__inference_model_layer_call_fn_9900
$__inference_model_layer_call_fn_9415
$__inference_model_layer_call_fn_9859
$__inference_model_layer_call_fn_9321ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ё2џ
D__inference_sequential_layer_call_and_return_conditional_losses_9924
D__inference_sequential_layer_call_and_return_conditional_losses_9912
D__inference_sequential_layer_call_and_return_conditional_losses_8045
D__inference_sequential_layer_call_and_return_conditional_losses_8055ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
т2п
)__inference_sequential_layer_call_fn_9942
)__inference_sequential_layer_call_fn_8094
)__inference_sequential_layer_call_fn_8075
)__inference_sequential_layer_call_fn_9933ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_sequential_1_layer_call_and_return_conditional_losses_8149
F__inference_sequential_1_layer_call_and_return_conditional_losses_9954
F__inference_sequential_1_layer_call_and_return_conditional_losses_8139
F__inference_sequential_1_layer_call_and_return_conditional_losses_9966ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ъ2ч
+__inference_sequential_1_layer_call_fn_8188
+__inference_sequential_1_layer_call_fn_9975
+__inference_sequential_1_layer_call_fn_8169
+__inference_sequential_1_layer_call_fn_9984ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
F__inference_sequential_2_layer_call_and_return_conditional_losses_8243
G__inference_sequential_2_layer_call_and_return_conditional_losses_10008
F__inference_sequential_2_layer_call_and_return_conditional_losses_8233
F__inference_sequential_2_layer_call_and_return_conditional_losses_9996ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ь2щ
+__inference_sequential_2_layer_call_fn_8263
,__inference_sequential_2_layer_call_fn_10017
,__inference_sequential_2_layer_call_fn_10026
+__inference_sequential_2_layer_call_fn_8282ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
и2е
G__inference_sequential_3_layer_call_and_return_conditional_losses_10050
G__inference_sequential_3_layer_call_and_return_conditional_losses_10038
F__inference_sequential_3_layer_call_and_return_conditional_losses_8327
F__inference_sequential_3_layer_call_and_return_conditional_losses_8337ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ь2щ
,__inference_sequential_3_layer_call_fn_10059
,__inference_sequential_3_layer_call_fn_10068
+__inference_sequential_3_layer_call_fn_8376
+__inference_sequential_3_layer_call_fn_8357ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
и2е
F__inference_sequential_4_layer_call_and_return_conditional_losses_8461
G__inference_sequential_4_layer_call_and_return_conditional_losses_10132
F__inference_sequential_4_layer_call_and_return_conditional_losses_8451
G__inference_sequential_4_layer_call_and_return_conditional_losses_10100ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ь2щ
+__inference_sequential_4_layer_call_fn_8500
+__inference_sequential_4_layer_call_fn_8481
,__inference_sequential_4_layer_call_fn_10141
,__inference_sequential_4_layer_call_fn_10150ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
F__inference_concatenate_layer_call_and_return_conditional_losses_10196
F__inference_concatenate_layer_call_and_return_conditional_losses_10183
F__inference_concatenate_layer_call_and_return_conditional_losses_10170
F__inference_concatenate_layer_call_and_return_conditional_losses_10157Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
№2ў
+__inference_concatenate_layer_call_fn_10163
+__inference_concatenate_layer_call_fn_10189
+__inference_concatenate_layer_call_fn_10176
+__inference_concatenate_layer_call_fn_10202Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2е
G__inference_sequential_5_layer_call_and_return_conditional_losses_10266
F__inference_sequential_5_layer_call_and_return_conditional_losses_8585
G__inference_sequential_5_layer_call_and_return_conditional_losses_10234
F__inference_sequential_5_layer_call_and_return_conditional_losses_8575ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ь2щ
,__inference_sequential_5_layer_call_fn_10275
+__inference_sequential_5_layer_call_fn_8624
,__inference_sequential_5_layer_call_fn_10284
+__inference_sequential_5_layer_call_fn_8605ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
и2е
G__inference_sequential_6_layer_call_and_return_conditional_losses_10348
G__inference_sequential_6_layer_call_and_return_conditional_losses_10316
F__inference_sequential_6_layer_call_and_return_conditional_losses_8699
F__inference_sequential_6_layer_call_and_return_conditional_losses_8709ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ь2щ
+__inference_sequential_6_layer_call_fn_8748
+__inference_sequential_6_layer_call_fn_8729
,__inference_sequential_6_layer_call_fn_10357
,__inference_sequential_6_layer_call_fn_10366ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
и2е
G__inference_sequential_7_layer_call_and_return_conditional_losses_10398
F__inference_sequential_7_layer_call_and_return_conditional_losses_8833
G__inference_sequential_7_layer_call_and_return_conditional_losses_10430
F__inference_sequential_7_layer_call_and_return_conditional_losses_8823ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ь2щ
,__inference_sequential_7_layer_call_fn_10448
,__inference_sequential_7_layer_call_fn_10439
+__inference_sequential_7_layer_call_fn_8872
+__inference_sequential_7_layer_call_fn_8853ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ђ2®
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_8911„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€A
Р2Н
1__inference_conv2d_transpose_4_layer_call_fn_8921„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€A
…B∆
"__inference_signature_wrapper_9466input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_conv2d_layer_call_and_return_conditional_losses_10459Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_conv2d_layer_call_fn_10468Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѓ2ђ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8006а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_max_pooling2d_layer_call_fn_8012а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
н2к
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10479Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_conv2d_1_layer_call_fn_10488Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
±2Ѓ
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8100а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_max_pooling2d_1_layer_call_fn_8106а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
н2к
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10499Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_conv2d_2_layer_call_fn_10508Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
±2Ѓ
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8194а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_max_pooling2d_2_layer_call_fn_8200а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
н2к
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10519Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_conv2d_3_layer_call_fn_10528Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
±2Ѓ
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8288а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_max_pooling2d_3_layer_call_fn_8294а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
©2¶
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8411„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
О2Л
/__inference_conv2d_transpose_layer_call_fn_8421„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
ѓ2ђ
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8434а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
,__inference_up_sampling2d_layer_call_fn_8440а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ2®
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8535„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Р2Н
1__inference_conv2d_transpose_1_layer_call_fn_8545„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
±2Ѓ
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8558а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_up_sampling2d_1_layer_call_fn_8564а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ2®
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_8659„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Р2Н
1__inference_conv2d_transpose_2_layer_call_fn_8669„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
±2Ѓ
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8682а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_up_sampling2d_2_layer_call_fn_8688а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ2®
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8783„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
Р2Н
1__inference_conv2d_transpose_3_layer_call_fn_8793„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
±2Ѓ
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8806а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_up_sampling2d_3_layer_call_fn_8812а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€«
__inference__wrapped_model_8000£QRSTUVWXYZ[\]^_`FG:Ґ7
0Ґ-
+К(
input_1€€€€€€€€€АА
™ "Q™N
L
conv2d_transpose_46К3
conv2d_transpose_4€€€€€€€€€ААш
F__inference_concatenate_layer_call_and_return_conditional_losses_10157≠|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
*К'
inputs/1€€€€€€€€€@@@
™ "-Ґ*
#К 
0€€€€€€€€€@@`
Ъ ш
F__inference_concatenate_layer_call_and_return_conditional_losses_10170≠|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*К'
inputs/1€€€€€€€€€   
™ "-Ґ*
#К 
0€€€€€€€€€  0
Ъ ь
F__inference_concatenate_layer_call_and_return_conditional_losses_10183±~Ґ{
tҐq
oЪl
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
,К)
inputs/1€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€ААA
Ъ ш
F__inference_concatenate_layer_call_and_return_conditional_losses_10196≠|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*К'
inputs/1€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ –
+__inference_concatenate_layer_call_fn_10163†|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
*К'
inputs/1€€€€€€€€€@@@
™ " К€€€€€€€€€@@`–
+__inference_concatenate_layer_call_fn_10176†|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*К'
inputs/1€€€€€€€€€   
™ " К€€€€€€€€€  0‘
+__inference_concatenate_layer_call_fn_10189§~Ґ{
tҐq
oЪl
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
,К)
inputs/1€€€€€€€€€АА
™ ""К€€€€€€€€€ААA–
+__inference_concatenate_layer_call_fn_10202†|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
*К'
inputs/1€€€€€€€€€
™ " К€€€€€€€€€≥
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10479lST7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@@
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ Л
(__inference_conv2d_1_layer_call_fn_10488_ST7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@@
™ " К€€€€€€€€€@@ ≥
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10499lUV7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€   
™ "-Ґ*
#К 
0€€€€€€€€€  
Ъ Л
(__inference_conv2d_2_layer_call_fn_10508_UV7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€   
™ " К€€€€€€€€€  ≥
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10519lWX7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Л
(__inference_conv2d_3_layer_call_fn_10528_WX7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€µ
A__inference_conv2d_layer_call_and_return_conditional_losses_10459pQR9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА@
Ъ Н
&__inference_conv2d_layer_call_fn_10468cQR9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€АА@б
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8535Р[\IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
1__inference_conv2d_transpose_1_layer_call_fn_8545Г[\IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€б
L__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_8659Р]^IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ є
1__inference_conv2d_transpose_2_layer_call_fn_8669Г]^IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ б
L__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_8783Р_`IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ є
1__inference_conv2d_transpose_3_layer_call_fn_8793Г_`IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@б
L__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_8911РFGIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€A
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ є
1__inference_conv2d_transpose_4_layer_call_fn_8921ГFGIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€A
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€я
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8411РYZIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
/__inference_conv2d_transpose_layer_call_fn_8421ГYZIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8100ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_1_layer_call_fn_8106СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8194ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_2_layer_call_fn_8200СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_8288ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_3_layer_call_fn_8294СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€к
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8006ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ¬
,__inference_max_pooling2d_layer_call_fn_8012СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ё
?__inference_model_layer_call_and_return_conditional_losses_9173ЩQRSTUVWXYZ[\]^_`FGBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ё
?__inference_model_layer_call_and_return_conditional_losses_9226ЩQRSTUVWXYZ[\]^_`FGBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ћ
?__inference_model_layer_call_and_return_conditional_losses_9642ИQRSTUVWXYZ[\]^_`FGAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ ћ
?__inference_model_layer_call_and_return_conditional_losses_9818ИQRSTUVWXYZ[\]^_`FGAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ µ
$__inference_model_layer_call_fn_9321МQRSTUVWXYZ[\]^_`FGBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€µ
$__inference_model_layer_call_fn_9415МQRSTUVWXYZ[\]^_`FGBҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€і
$__inference_model_layer_call_fn_9859ЛQRSTUVWXYZ[\]^_`FGAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€і
$__inference_model_layer_call_fn_9900ЛQRSTUVWXYZ[\]^_`FGAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€∆
F__inference_sequential_1_layer_call_and_return_conditional_losses_8139|STGҐD
=Ґ:
0К-
conv2d_1_input€€€€€€€€€@@@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ ∆
F__inference_sequential_1_layer_call_and_return_conditional_losses_8149|STGҐD
=Ґ:
0К-
conv2d_1_input€€€€€€€€€@@@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ Њ
F__inference_sequential_1_layer_call_and_return_conditional_losses_9954tST?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ Њ
F__inference_sequential_1_layer_call_and_return_conditional_losses_9966tST?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ Ю
+__inference_sequential_1_layer_call_fn_8169oSTGҐD
=Ґ:
0К-
conv2d_1_input€€€€€€€€€@@@
p

 
™ " К€€€€€€€€€   Ю
+__inference_sequential_1_layer_call_fn_8188oSTGҐD
=Ґ:
0К-
conv2d_1_input€€€€€€€€€@@@
p 

 
™ " К€€€€€€€€€   Ц
+__inference_sequential_1_layer_call_fn_9975gST?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@@
p

 
™ " К€€€€€€€€€   Ц
+__inference_sequential_1_layer_call_fn_9984gST?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@@
p 

 
™ " К€€€€€€€€€   њ
G__inference_sequential_2_layer_call_and_return_conditional_losses_10008tUV?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€   
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ∆
F__inference_sequential_2_layer_call_and_return_conditional_losses_8233|UVGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€   
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ∆
F__inference_sequential_2_layer_call_and_return_conditional_losses_8243|UVGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€   
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Њ
F__inference_sequential_2_layer_call_and_return_conditional_losses_9996tUV?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€   
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Ч
,__inference_sequential_2_layer_call_fn_10017gUV?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€   
p

 
™ " К€€€€€€€€€Ч
,__inference_sequential_2_layer_call_fn_10026gUV?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€   
p 

 
™ " К€€€€€€€€€Ю
+__inference_sequential_2_layer_call_fn_8263oUVGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€   
p

 
™ " К€€€€€€€€€Ю
+__inference_sequential_2_layer_call_fn_8282oUVGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€   
p 

 
™ " К€€€€€€€€€њ
G__inference_sequential_3_layer_call_and_return_conditional_losses_10038tWX?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ њ
G__inference_sequential_3_layer_call_and_return_conditional_losses_10050tWX?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ∆
F__inference_sequential_3_layer_call_and_return_conditional_losses_8327|WXGҐD
=Ґ:
0К-
conv2d_3_input€€€€€€€€€
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ∆
F__inference_sequential_3_layer_call_and_return_conditional_losses_8337|WXGҐD
=Ґ:
0К-
conv2d_3_input€€€€€€€€€
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Ч
,__inference_sequential_3_layer_call_fn_10059gWX?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ " К€€€€€€€€€Ч
,__inference_sequential_3_layer_call_fn_10068gWX?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ " К€€€€€€€€€Ю
+__inference_sequential_3_layer_call_fn_8357oWXGҐD
=Ґ:
0К-
conv2d_3_input€€€€€€€€€
p

 
™ " К€€€€€€€€€Ю
+__inference_sequential_3_layer_call_fn_8376oWXGҐD
=Ґ:
0К-
conv2d_3_input€€€€€€€€€
p 

 
™ " К€€€€€€€€€њ
G__inference_sequential_4_layer_call_and_return_conditional_losses_10100tYZ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ њ
G__inference_sequential_4_layer_call_and_return_conditional_losses_10132tYZ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ б
F__inference_sequential_4_layer_call_and_return_conditional_losses_8451ЦYZOҐL
EҐB
8К5
conv2d_transpose_input€€€€€€€€€
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
F__inference_sequential_4_layer_call_and_return_conditional_losses_8461ЦYZOҐL
EҐB
8К5
conv2d_transpose_input€€€€€€€€€
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ©
,__inference_sequential_4_layer_call_fn_10141yYZ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€©
,__inference_sequential_4_layer_call_fn_10150yYZ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€є
+__inference_sequential_4_layer_call_fn_8481ЙYZOҐL
EҐB
8К5
conv2d_transpose_input€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€є
+__inference_sequential_4_layer_call_fn_8500ЙYZOҐL
EҐB
8К5
conv2d_transpose_input€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€њ
G__inference_sequential_5_layer_call_and_return_conditional_losses_10234t[\?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "-Ґ*
#К 
0€€€€€€€€€  
Ъ њ
G__inference_sequential_5_layer_call_and_return_conditional_losses_10266t[\?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€  
Ъ г
F__inference_sequential_5_layer_call_and_return_conditional_losses_8575Ш[\QҐN
GҐD
:К7
conv2d_transpose_1_input€€€€€€€€€
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ г
F__inference_sequential_5_layer_call_and_return_conditional_losses_8585Ш[\QҐN
GҐD
:К7
conv2d_transpose_1_input€€€€€€€€€
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ©
,__inference_sequential_5_layer_call_fn_10275y[\?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€©
,__inference_sequential_5_layer_call_fn_10284y[\?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ї
+__inference_sequential_5_layer_call_fn_8605Л[\QҐN
GҐD
:К7
conv2d_transpose_1_input€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ї
+__inference_sequential_5_layer_call_fn_8624Л[\QҐN
GҐD
:К7
conv2d_transpose_1_input€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€њ
G__inference_sequential_6_layer_call_and_return_conditional_losses_10316t]^?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  0
p

 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ њ
G__inference_sequential_6_layer_call_and_return_conditional_losses_10348t]^?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  0
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ г
F__inference_sequential_6_layer_call_and_return_conditional_losses_8699Ш]^QҐN
GҐD
:К7
conv2d_transpose_2_input€€€€€€€€€  0
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ г
F__inference_sequential_6_layer_call_and_return_conditional_losses_8709Ш]^QҐN
GҐD
:К7
conv2d_transpose_2_input€€€€€€€€€  0
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ©
,__inference_sequential_6_layer_call_fn_10357y]^?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  0
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ©
,__inference_sequential_6_layer_call_fn_10366y]^?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  0
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ї
+__inference_sequential_6_layer_call_fn_8729Л]^QҐN
GҐD
:К7
conv2d_transpose_2_input€€€€€€€€€  0
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ї
+__inference_sequential_6_layer_call_fn_8748Л]^QҐN
GҐD
:К7
conv2d_transpose_2_input€€€€€€€€€  0
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Ѕ
G__inference_sequential_7_layer_call_and_return_conditional_losses_10398v_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@`
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА@
Ъ Ѕ
G__inference_sequential_7_layer_call_and_return_conditional_losses_10430v_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@`
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА@
Ъ г
F__inference_sequential_7_layer_call_and_return_conditional_losses_8823Ш_`QҐN
GҐD
:К7
conv2d_transpose_3_input€€€€€€€€€@@`
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ г
F__inference_sequential_7_layer_call_and_return_conditional_losses_8833Ш_`QҐN
GҐD
:К7
conv2d_transpose_3_input€€€€€€€€€@@`
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ©
,__inference_sequential_7_layer_call_fn_10439y_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@`
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@©
,__inference_sequential_7_layer_call_fn_10448y_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@`
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ї
+__inference_sequential_7_layer_call_fn_8853Л_`QҐN
GҐD
:К7
conv2d_transpose_3_input€€€€€€€€€@@`
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ї
+__inference_sequential_7_layer_call_fn_8872Л_`QҐN
GҐD
:К7
conv2d_transpose_3_input€€€€€€€€€@@`
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ƒ
D__inference_sequential_layer_call_and_return_conditional_losses_8045|QRGҐD
=Ґ:
0К-
conv2d_input€€€€€€€€€АА
p

 
™ "-Ґ*
#К 
0€€€€€€€€€@@@
Ъ ƒ
D__inference_sequential_layer_call_and_return_conditional_losses_8055|QRGҐD
=Ґ:
0К-
conv2d_input€€€€€€€€€АА
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€@@@
Ъ Њ
D__inference_sequential_layer_call_and_return_conditional_losses_9912vQRAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "-Ґ*
#К 
0€€€€€€€€€@@@
Ъ Њ
D__inference_sequential_layer_call_and_return_conditional_losses_9924vQRAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€@@@
Ъ Ь
)__inference_sequential_layer_call_fn_8075oQRGҐD
=Ґ:
0К-
conv2d_input€€€€€€€€€АА
p

 
™ " К€€€€€€€€€@@@Ь
)__inference_sequential_layer_call_fn_8094oQRGҐD
=Ґ:
0К-
conv2d_input€€€€€€€€€АА
p 

 
™ " К€€€€€€€€€@@@Ц
)__inference_sequential_layer_call_fn_9933iQRAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ " К€€€€€€€€€@@@Ц
)__inference_sequential_layer_call_fn_9942iQRAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ " К€€€€€€€€€@@@’
"__inference_signature_wrapper_9466ЃQRSTUVWXYZ[\]^_`FGEҐB
Ґ 
;™8
6
input_1+К(
input_1€€€€€€€€€АА"Q™N
L
conv2d_transpose_46К3
conv2d_transpose_4€€€€€€€€€ААм
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_8558ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_up_sampling2d_1_layer_call_fn_8564СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_8682ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_up_sampling2d_2_layer_call_fn_8688СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_8806ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_up_sampling2d_3_layer_call_fn_8812СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€к
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_8434ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ¬
,__inference_up_sampling2d_layer_call_fn_8440СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€