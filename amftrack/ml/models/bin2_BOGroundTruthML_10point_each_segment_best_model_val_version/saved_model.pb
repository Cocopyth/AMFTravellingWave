��
��
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��
r
output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias/v
k
!output/bias/v/Read/ReadVariableOpReadVariableOpoutput/bias/v*
_output_shapes
:*
dtype0
{
output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_nameoutput/kernel/v
t
#output/kernel/v/Read/ReadVariableOpReadVariableOpoutput/kernel/v*
_output_shapes
:	�*
dtype0
u
dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias/v
n
"dense_1/bias/v/Read/ReadVariableOpReadVariableOpdense_1/bias/v*
_output_shapes	
:�*
dtype0
~
dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�w�*!
shared_namedense_1/kernel/v
w
$dense_1/kernel/v/Read/ReadVariableOpReadVariableOpdense_1/kernel/v* 
_output_shapes
:
�w�*
dtype0
�
batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namebatch_normalization/beta/v
�
.batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/v*
_output_shapes
:`*
dtype0
�
batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization/gamma/v
�
/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/v*
_output_shapes
:`*
dtype0
r
conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv_3/bias/v
k
!conv_3/bias/v/Read/ReadVariableOpReadVariableOpconv_3/bias/v*
_output_shapes
:`*
dtype0

conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�`* 
shared_nameconv_3/kernel/v
x
#conv_3/kernel/v/Read/ReadVariableOpReadVariableOpconv_3/kernel/v*#
_output_shapes
:�`*
dtype0
s
conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv_2/bias/v
l
!conv_2/bias/v/Read/ReadVariableOpReadVariableOpconv_2/bias/v*
_output_shapes	
:�*
dtype0

conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv_2/kernel/v
x
#conv_2/kernel/v/Read/ReadVariableOpReadVariableOpconv_2/kernel/v*#
_output_shapes
:@�*
dtype0
r
conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_1/bias/v
k
!conv_1/bias/v/Read/ReadVariableOpReadVariableOpconv_1/bias/v*
_output_shapes
:@*
dtype0
~
conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv_1/kernel/v
w
#conv_1/kernel/v/Read/ReadVariableOpReadVariableOpconv_1/kernel/v*"
_output_shapes
:@*
dtype0
r
output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias/m
k
!output/bias/m/Read/ReadVariableOpReadVariableOpoutput/bias/m*
_output_shapes
:*
dtype0
{
output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_nameoutput/kernel/m
t
#output/kernel/m/Read/ReadVariableOpReadVariableOpoutput/kernel/m*
_output_shapes
:	�*
dtype0
u
dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias/m
n
"dense_1/bias/m/Read/ReadVariableOpReadVariableOpdense_1/bias/m*
_output_shapes	
:�*
dtype0
~
dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�w�*!
shared_namedense_1/kernel/m
w
$dense_1/kernel/m/Read/ReadVariableOpReadVariableOpdense_1/kernel/m* 
_output_shapes
:
�w�*
dtype0
�
batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namebatch_normalization/beta/m
�
.batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/m*
_output_shapes
:`*
dtype0
�
batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization/gamma/m
�
/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/m*
_output_shapes
:`*
dtype0
r
conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv_3/bias/m
k
!conv_3/bias/m/Read/ReadVariableOpReadVariableOpconv_3/bias/m*
_output_shapes
:`*
dtype0

conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�`* 
shared_nameconv_3/kernel/m
x
#conv_3/kernel/m/Read/ReadVariableOpReadVariableOpconv_3/kernel/m*#
_output_shapes
:�`*
dtype0
s
conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv_2/bias/m
l
!conv_2/bias/m/Read/ReadVariableOpReadVariableOpconv_2/bias/m*
_output_shapes	
:�*
dtype0

conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv_2/kernel/m
x
#conv_2/kernel/m/Read/ReadVariableOpReadVariableOpconv_2/kernel/m*#
_output_shapes
:@�*
dtype0
r
conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_1/bias/m
k
!conv_1/bias/m/Read/ReadVariableOpReadVariableOpconv_1/bias/m*
_output_shapes
:@*
dtype0
~
conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv_1/kernel/m
w
#conv_1/kernel/m/Read/ReadVariableOpReadVariableOpconv_1/kernel/m*"
_output_shapes
:@*
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
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�w�*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
�w�*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:`*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:`*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:`*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:`*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:`*
dtype0
{
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�`*
shared_nameconv_3/kernel
t
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*#
_output_shapes
:�`*
dtype0
o
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv_2/bias
h
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes	
:�*
dtype0
{
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_nameconv_2/kernel
t
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*#
_output_shapes
:@�*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:@*
dtype0
z
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_1/kernel
s
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*"
_output_shapes
:@*
dtype0
�
serving_default_input_1Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3112711

NoOpNoOp
�e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�d
value�dB�d B�d
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias
 "_jit_compiled_convolution_op*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator* 
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[_random_generator* 
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias*
j
 0
!1
)2
*3
24
35
<6
=7
>8
?9
S10
T11
b12
c13*
Z
 0
!1
)2
*3
24
35
<6
=7
S8
T9
b10
c11*
:
d0
e1
f2
g3
h4
i5
j6
k7* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
qtrace_0
rtrace_1
strace_2
ttrace_3* 
6
utrace_0
vtrace_1
wtrace_2
xtrace_3* 
* 
� m�!m�)m�*m�2m�3m�<m�=m�Sm�Tm�bm�cm� v�!v�)v�*v�2v�3v�<v�=v�Sv�Tv�bv�cv�*

yserving_default* 
* 
* 
* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0* 

�trace_0* 

 0
!1*

 0
!1*

d0
e1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
�activity_regularizer_fn
*&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

)0
*1*

)0
*1*

f0
g1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
�activity_regularizer_fn
*(&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

20
31*

20
31*

h0
i1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
�activity_regularizer_fn
*1&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
<0
=1
>2
?3*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

S0
T1*

S0
T1*

j0
k1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
�activity_regularizer_fn
*R&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

b0
c1*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

>0
?1*
J
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
9*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

d0
e1* 
* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 

f0
g1* 
* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 

h0
i1* 
* 

�trace_0* 

�trace_0* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

j0
k1* 
* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
{u
VARIABLE_VALUEconv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEbatch_normalization/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEbatch_normalization/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEoutput/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEoutput/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEbatch_normalization/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEbatch_normalization/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEoutput/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEoutput/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#conv_1/kernel/m/Read/ReadVariableOp!conv_1/bias/m/Read/ReadVariableOp#conv_2/kernel/m/Read/ReadVariableOp!conv_2/bias/m/Read/ReadVariableOp#conv_3/kernel/m/Read/ReadVariableOp!conv_3/bias/m/Read/ReadVariableOp/batch_normalization/gamma/m/Read/ReadVariableOp.batch_normalization/beta/m/Read/ReadVariableOp$dense_1/kernel/m/Read/ReadVariableOp"dense_1/bias/m/Read/ReadVariableOp#output/kernel/m/Read/ReadVariableOp!output/bias/m/Read/ReadVariableOp#conv_1/kernel/v/Read/ReadVariableOp!conv_1/bias/v/Read/ReadVariableOp#conv_2/kernel/v/Read/ReadVariableOp!conv_2/bias/v/Read/ReadVariableOp#conv_3/kernel/v/Read/ReadVariableOp!conv_3/bias/v/Read/ReadVariableOp/batch_normalization/gamma/v/Read/ReadVariableOp.batch_normalization/beta/v/Read/ReadVariableOp$dense_1/kernel/v/Read/ReadVariableOp"dense_1/bias/v/Read/ReadVariableOp#output/kernel/v/Read/ReadVariableOp!output/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_3113819
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_1/kerneldense_1/biasoutput/kerneloutput/biastotal_1count_1totalcountconv_1/kernel/mconv_1/bias/mconv_2/kernel/mconv_2/bias/mconv_3/kernel/mconv_3/bias/mbatch_normalization/gamma/mbatch_normalization/beta/mdense_1/kernel/mdense_1/bias/moutput/kernel/moutput/bias/mconv_1/kernel/vconv_1/bias/vconv_2/kernel/vconv_2/bias/vconv_3/kernel/vconv_3/bias/vbatch_normalization/gamma/vbatch_normalization/beta/vdense_1/kernel/vdense_1/bias/voutput/kernel/voutput/bias/v*6
Tin/
-2+*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_3113955��
�	
�
C__inference_output_layer_call_and_return_conditional_losses_3111889

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113563

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_rescaling_layer_call_fn_3113222

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_3111683e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_3113955
file_prefix4
assignvariableop_conv_1_kernel:@,
assignvariableop_1_conv_1_bias:@7
 assignvariableop_2_conv_2_kernel:@�-
assignvariableop_3_conv_2_bias:	�7
 assignvariableop_4_conv_3_kernel:�`,
assignvariableop_5_conv_3_bias:`:
,assignvariableop_6_batch_normalization_gamma:`9
+assignvariableop_7_batch_normalization_beta:`@
2assignvariableop_8_batch_normalization_moving_mean:`D
6assignvariableop_9_batch_normalization_moving_variance:`6
"assignvariableop_10_dense_1_kernel:
�w�/
 assignvariableop_11_dense_1_bias:	�4
!assignvariableop_12_output_kernel:	�-
assignvariableop_13_output_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 9
#assignvariableop_18_conv_1_kernel_m:@/
!assignvariableop_19_conv_1_bias_m:@:
#assignvariableop_20_conv_2_kernel_m:@�0
!assignvariableop_21_conv_2_bias_m:	�:
#assignvariableop_22_conv_3_kernel_m:�`/
!assignvariableop_23_conv_3_bias_m:`=
/assignvariableop_24_batch_normalization_gamma_m:`<
.assignvariableop_25_batch_normalization_beta_m:`8
$assignvariableop_26_dense_1_kernel_m:
�w�1
"assignvariableop_27_dense_1_bias_m:	�6
#assignvariableop_28_output_kernel_m:	�/
!assignvariableop_29_output_bias_m:9
#assignvariableop_30_conv_1_kernel_v:@/
!assignvariableop_31_conv_1_bias_v:@:
#assignvariableop_32_conv_2_kernel_v:@�0
!assignvariableop_33_conv_2_bias_v:	�:
#assignvariableop_34_conv_3_kernel_v:�`/
!assignvariableop_35_conv_3_bias_v:`=
/assignvariableop_36_batch_normalization_gamma_v:`<
.assignvariableop_37_batch_normalization_beta_v:`8
$assignvariableop_38_dense_1_kernel_v:
�w�1
"assignvariableop_39_dense_1_bias_v:	�6
#assignvariableop_40_output_kernel_v:	�/
!assignvariableop_41_output_bias_v:
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv_1_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv_1_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv_2_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv_2_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv_3_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv_3_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_gamma_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_beta_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_dense_1_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp"assignvariableop_27_dense_1_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_output_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_output_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv_1_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv_1_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv_2_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp!assignvariableop_33_conv_2_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_conv_3_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_conv_3_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_gamma_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_beta_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp$assignvariableop_38_dense_1_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp"assignvariableop_39_dense_1_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_output_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_output_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

c
D__inference_dropout_layer_call_and_return_conditional_losses_3113482

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������`*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������`t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������`n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������`^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
F
/__inference_conv_2_activity_regularizer_3111559
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
�
__inference_loss_fn_6_3113659J
6dense_1_kernel_regularizer_abs_readvariableop_resource:
�w�
identity��-dense_1/kernel/Regularizer/Abs/ReadVariableOp�
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp
�
�
H__inference_dense_1_layer_call_and_return_all_conditional_losses_3113513

inputs
unknown:
�w�
	unknown_0:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858�
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *9
f4R2
0__inference_dense_1_activity_regularizer_3111667p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������w: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������w
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_3112376
input_1
unknown:@
	unknown_0:@ 
	unknown_1:@�
	unknown_2:	� 
	unknown_3:�`
	unknown_4:`
	unknown_5:`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:
�w�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3112304o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
__inference_loss_fn_0_3113593K
5conv_1_kernel_regularizer_abs_readvariableop_resource:@
identity��,conv_1/kernel/Regularizer/Abs/ReadVariableOp�
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_3111877

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv_1_layer_call_and_return_conditional_losses_3113279

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@�
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_1_layer_call_fn_3113502

inputs
unknown:
�w�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������w: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������w
 
_user_specified_nameinputs
��
�	
G__inference_sequential_layer_call_and_return_conditional_losses_3112502
input_1$
conv_1_3112380:@
conv_1_3112382:@%
conv_2_3112393:@�
conv_2_3112395:	�%
conv_3_3112406:�`
conv_3_3112408:`)
batch_normalization_3112419:`)
batch_normalization_3112421:`)
batch_normalization_3112423:`)
batch_normalization_3112425:`#
dense_1_3112430:
�w�
dense_1_3112432:	�!
output_3112444:	�
output_3112446:
identity

identity_1

identity_2

identity_3

identity_4��+batch_normalization/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp�conv_2/StatefulPartitionedCall�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp�conv_3/StatefulPartitionedCall�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp�dense_1/StatefulPartitionedCall�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�output/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_3111683�
conv_1/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv_1_3112380conv_1_3112382*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713�
*conv_1/ActivityRegularizer/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_1_activity_regularizer_3111546w
 conv_1/ActivityRegularizer/ShapeShape'conv_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_1/ActivityRegularizer/strided_sliceStridedSlice)conv_1/ActivityRegularizer/Shape:output:07conv_1/ActivityRegularizer/strided_slice/stack:output:09conv_1/ActivityRegularizer/strided_slice/stack_1:output:09conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_1/ActivityRegularizer/CastCast1conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_1/ActivityRegularizer/truedivRealDiv3conv_1/ActivityRegularizer/PartitionedCall:output:0#conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_3112393conv_2_3112395*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755�
*conv_2/ActivityRegularizer/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_2_activity_regularizer_3111559w
 conv_2/ActivityRegularizer/ShapeShape'conv_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_2/ActivityRegularizer/strided_sliceStridedSlice)conv_2/ActivityRegularizer/Shape:output:07conv_2/ActivityRegularizer/strided_slice/stack:output:09conv_2/ActivityRegularizer/strided_slice/stack_1:output:09conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_2/ActivityRegularizer/CastCast1conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_2/ActivityRegularizer/truedivRealDiv3conv_2/ActivityRegularizer/PartitionedCall:output:0#conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_3112406conv_3_3112408*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797�
*conv_3/ActivityRegularizer/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_3_activity_regularizer_3111572w
 conv_3/ActivityRegularizer/ShapeShape'conv_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_3/ActivityRegularizer/strided_sliceStridedSlice)conv_3/ActivityRegularizer/Shape:output:07conv_3/ActivityRegularizer/strided_slice/stack:output:09conv_3/ActivityRegularizer/strided_slice/stack_1:output:09conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_3/ActivityRegularizer/CastCast1conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_3/ActivityRegularizer/truedivRealDiv3conv_3/ActivityRegularizer/PartitionedCall:output:0#conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_3112419batch_normalization_3112421batch_normalization_3112423batch_normalization_3112425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111596�
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3111825�
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������w* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3111833�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_3112430dense_1_3112432*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858�
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *9
f4R2
0__inference_dense_1_activity_regularizer_3111667y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3111877�
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_3112444output_3112446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_3111889
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3112380*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3112382*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3112393*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3112395*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3112406*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3112408*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3112430* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3112432*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������f

Identity_1Identity&conv_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&conv_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_3Identity&conv_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: g

Identity_4Identity'dense_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv_1/StatefulPartitionedCall+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp^conv_2/StatefulPartitionedCall+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp^conv_3/StatefulPartitionedCall+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp ^dense_1/StatefulPartitionedCall,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
b
F__inference_rescaling_layer_call_and_return_conditional_losses_3111683

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������Q
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
mulMulinputsCast/x:output:0*
T0*,
_output_shapes
:����������X
addAddV2mul:z:0
Cast_1:y:0*
T0*,
_output_shapes
:����������T
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
0__inference_dense_1_activity_regularizer_3111667
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
b
)__inference_dropout_layer_call_fn_3113465

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3112064t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111596

inputs/
!batchnorm_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`1
#batchnorm_readvariableop_1_resource:`1
#batchnorm_readvariableop_2_resource:`
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������`z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:`z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������`o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������`�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������`: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������`
 
_user_specified_nameinputs
�
�
G__inference_conv_2_layer_call_and_return_all_conditional_losses_3113299

inputs
unknown:@�
	unknown_0:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755�
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_2_activity_regularizer_3111559u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:�����������X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_3113637L
5conv_3_kernel_regularizer_abs_readvariableop_resource:�`
identity��,conv_3/kernel/Regularizer/Abs/ReadVariableOp�
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5conv_3_kernel_regularizer_abs_readvariableop_resource*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp
�%
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113455

inputs5
'assignmovingavg_readvariableop_resource:`7
)assignmovingavg_1_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`/
!batchnorm_readvariableop_resource:`
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:`*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:`�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������`s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:`*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:`x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:`~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������`h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:`v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������`o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������`�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������`: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������`
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_3111533
input_1S
=sequential_conv_1_conv1d_expanddims_1_readvariableop_resource:@?
1sequential_conv_1_biasadd_readvariableop_resource:@T
=sequential_conv_2_conv1d_expanddims_1_readvariableop_resource:@�@
1sequential_conv_2_biasadd_readvariableop_resource:	�T
=sequential_conv_3_conv1d_expanddims_1_readvariableop_resource:�`?
1sequential_conv_3_biasadd_readvariableop_resource:`N
@sequential_batch_normalization_batchnorm_readvariableop_resource:`R
Dsequential_batch_normalization_batchnorm_mul_readvariableop_resource:`P
Bsequential_batch_normalization_batchnorm_readvariableop_1_resource:`P
Bsequential_batch_normalization_batchnorm_readvariableop_2_resource:`E
1sequential_dense_1_matmul_readvariableop_resource:
�w�A
2sequential_dense_1_biasadd_readvariableop_resource:	�C
0sequential_output_matmul_readvariableop_resource:	�?
1sequential_output_biasadd_readvariableop_resource:
identity��7sequential/batch_normalization/batchnorm/ReadVariableOp�9sequential/batch_normalization/batchnorm/ReadVariableOp_1�9sequential/batch_normalization/batchnorm/ReadVariableOp_2�;sequential/batch_normalization/batchnorm/mul/ReadVariableOp�(sequential/conv_1/BiasAdd/ReadVariableOp�4sequential/conv_1/Conv1D/ExpandDims_1/ReadVariableOp�(sequential/conv_2/BiasAdd/ReadVariableOp�4sequential/conv_2/Conv1D/ExpandDims_1/ReadVariableOp�(sequential/conv_3/BiasAdd/ReadVariableOp�4sequential/conv_3/Conv1D/ExpandDims_1/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�(sequential/output/BiasAdd/ReadVariableOp�'sequential/output/MatMul/ReadVariableOp`
sequential/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <h
sequential/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������{
sequential/rescaling/Cast_1Cast&sequential/rescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
sequential/rescaling/mulMulinput_1$sequential/rescaling/Cast/x:output:0*
T0*,
_output_shapes
:�����������
sequential/rescaling/addAddV2sequential/rescaling/mul:z:0sequential/rescaling/Cast_1:y:0*
T0*,
_output_shapes
:����������r
'sequential/conv_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential/conv_1/Conv1D/ExpandDims
ExpandDimssequential/rescaling/add:z:00sequential/conv_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
4sequential/conv_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0k
)sequential/conv_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/conv_1/Conv1D/ExpandDims_1
ExpandDims<sequential/conv_1/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@�
sequential/conv_1/Conv1DConv2D,sequential/conv_1/Conv1D/ExpandDims:output:0.sequential/conv_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
�
 sequential/conv_1/Conv1D/SqueezeSqueeze!sequential/conv_1/Conv1D:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

����������
(sequential/conv_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/conv_1/BiasAddBiasAdd)sequential/conv_1/Conv1D/Squeeze:output:00sequential/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@y
sequential/conv_1/ReluRelu"sequential/conv_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@�
)sequential/conv_1/ActivityRegularizer/AbsAbs$sequential/conv_1/Relu:activations:0*
T0*,
_output_shapes
:����������@�
+sequential/conv_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
)sequential/conv_1/ActivityRegularizer/SumSum-sequential/conv_1/ActivityRegularizer/Abs:y:04sequential/conv_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: p
+sequential/conv_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
)sequential/conv_1/ActivityRegularizer/mulMul4sequential/conv_1/ActivityRegularizer/mul/x:output:02sequential/conv_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
+sequential/conv_1/ActivityRegularizer/ShapeShape$sequential/conv_1/Relu:activations:0*
T0*
_output_shapes
:�
9sequential/conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;sequential/conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/conv_1/ActivityRegularizer/strided_sliceStridedSlice4sequential/conv_1/ActivityRegularizer/Shape:output:0Bsequential/conv_1/ActivityRegularizer/strided_slice/stack:output:0Dsequential/conv_1/ActivityRegularizer/strided_slice/stack_1:output:0Dsequential/conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
*sequential/conv_1/ActivityRegularizer/CastCast<sequential/conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-sequential/conv_1/ActivityRegularizer/truedivRealDiv-sequential/conv_1/ActivityRegularizer/mul:z:0.sequential/conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: r
'sequential/conv_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential/conv_2/Conv1D/ExpandDims
ExpandDims$sequential/conv_1/Relu:activations:00sequential/conv_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@�
4sequential/conv_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0k
)sequential/conv_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/conv_2/Conv1D/ExpandDims_1
ExpandDims<sequential/conv_2/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
sequential/conv_2/Conv1DConv2D,sequential/conv_2/Conv1D/ExpandDims:output:0.sequential/conv_2/Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
�
 sequential/conv_2/Conv1D/SqueezeSqueeze!sequential/conv_2/Conv1D:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims

����������
(sequential/conv_2/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/conv_2/BiasAddBiasAdd)sequential/conv_2/Conv1D/Squeeze:output:00sequential/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������z
sequential/conv_2/ReluRelu"sequential/conv_2/BiasAdd:output:0*
T0*-
_output_shapes
:������������
)sequential/conv_2/ActivityRegularizer/AbsAbs$sequential/conv_2/Relu:activations:0*
T0*-
_output_shapes
:������������
+sequential/conv_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
)sequential/conv_2/ActivityRegularizer/SumSum-sequential/conv_2/ActivityRegularizer/Abs:y:04sequential/conv_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: p
+sequential/conv_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
)sequential/conv_2/ActivityRegularizer/mulMul4sequential/conv_2/ActivityRegularizer/mul/x:output:02sequential/conv_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
+sequential/conv_2/ActivityRegularizer/ShapeShape$sequential/conv_2/Relu:activations:0*
T0*
_output_shapes
:�
9sequential/conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;sequential/conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/conv_2/ActivityRegularizer/strided_sliceStridedSlice4sequential/conv_2/ActivityRegularizer/Shape:output:0Bsequential/conv_2/ActivityRegularizer/strided_slice/stack:output:0Dsequential/conv_2/ActivityRegularizer/strided_slice/stack_1:output:0Dsequential/conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
*sequential/conv_2/ActivityRegularizer/CastCast<sequential/conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-sequential/conv_2/ActivityRegularizer/truedivRealDiv-sequential/conv_2/ActivityRegularizer/mul:z:0.sequential/conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: r
'sequential/conv_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential/conv_3/Conv1D/ExpandDims
ExpandDims$sequential/conv_2/Relu:activations:00sequential/conv_3/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
4sequential/conv_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0k
)sequential/conv_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/conv_3/Conv1D/ExpandDims_1
ExpandDims<sequential/conv_3/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�`�
sequential/conv_3/Conv1DConv2D,sequential/conv_3/Conv1D/ExpandDims:output:0.sequential/conv_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������`*
paddingVALID*
strides
�
 sequential/conv_3/Conv1D/SqueezeSqueeze!sequential/conv_3/Conv1D:output:0*
T0*,
_output_shapes
:����������`*
squeeze_dims

����������
(sequential/conv_3/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
sequential/conv_3/BiasAddBiasAdd)sequential/conv_3/Conv1D/Squeeze:output:00sequential/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������`y
sequential/conv_3/ReluRelu"sequential/conv_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������`�
)sequential/conv_3/ActivityRegularizer/AbsAbs$sequential/conv_3/Relu:activations:0*
T0*,
_output_shapes
:����������`�
+sequential/conv_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
)sequential/conv_3/ActivityRegularizer/SumSum-sequential/conv_3/ActivityRegularizer/Abs:y:04sequential/conv_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: p
+sequential/conv_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
)sequential/conv_3/ActivityRegularizer/mulMul4sequential/conv_3/ActivityRegularizer/mul/x:output:02sequential/conv_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
+sequential/conv_3/ActivityRegularizer/ShapeShape$sequential/conv_3/Relu:activations:0*
T0*
_output_shapes
:�
9sequential/conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;sequential/conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential/conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3sequential/conv_3/ActivityRegularizer/strided_sliceStridedSlice4sequential/conv_3/ActivityRegularizer/Shape:output:0Bsequential/conv_3/ActivityRegularizer/strided_slice/stack:output:0Dsequential/conv_3/ActivityRegularizer/strided_slice/stack_1:output:0Dsequential/conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
*sequential/conv_3/ActivityRegularizer/CastCast<sequential/conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-sequential/conv_3/ActivityRegularizer/truedivRealDiv-sequential/conv_3/ActivityRegularizer/mul:z:0.sequential/conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
7sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOp@sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0s
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,sequential/batch_normalization/batchnorm/addAddV2?sequential/batch_normalization/batchnorm/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:`�
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:`�
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpDsequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0�
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0Csequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`�
.sequential/batch_normalization/batchnorm/mul_1Mul$sequential/conv_3/Relu:activations:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������`�
9sequential/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0�
.sequential/batch_normalization/batchnorm/mul_2MulAsequential/batch_normalization/batchnorm/ReadVariableOp_1:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:`�
9sequential/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0�
,sequential/batch_normalization/batchnorm/subSubAsequential/batch_normalization/batchnorm/ReadVariableOp_2:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:`�
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������`�
sequential/dropout/IdentityIdentity2sequential/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������`i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����;  �
sequential/flatten/ReshapeReshape$sequential/dropout/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:����������w�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
sequential/dense_1/MatMulMatMul#sequential/flatten/Reshape:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*sequential/dense_1/ActivityRegularizer/AbsAbs%sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:����������}
,sequential/dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
*sequential/dense_1/ActivityRegularizer/SumSum.sequential/dense_1/ActivityRegularizer/Abs:y:05sequential/dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: q
,sequential/dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
*sequential/dense_1/ActivityRegularizer/mulMul5sequential/dense_1/ActivityRegularizer/mul/x:output:03sequential/dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
,sequential/dense_1/ActivityRegularizer/ShapeShape%sequential/dense_1/Relu:activations:0*
T0*
_output_shapes
:�
:sequential/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
<sequential/dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
<sequential/dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
4sequential/dense_1/ActivityRegularizer/strided_sliceStridedSlice5sequential/dense_1/ActivityRegularizer/Shape:output:0Csequential/dense_1/ActivityRegularizer/strided_slice/stack:output:0Esequential/dense_1/ActivityRegularizer/strided_slice/stack_1:output:0Esequential/dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
+sequential/dense_1/ActivityRegularizer/CastCast=sequential/dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
.sequential/dense_1/ActivityRegularizer/truedivRealDiv.sequential/dense_1/ActivityRegularizer/mul:z:0/sequential/dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
sequential/dropout_1/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:�����������
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential/output/MatMulMatMul&sequential/dropout_1/Identity:output:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"sequential/output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp8^sequential/batch_normalization/batchnorm/ReadVariableOp:^sequential/batch_normalization/batchnorm/ReadVariableOp_1:^sequential/batch_normalization/batchnorm/ReadVariableOp_2<^sequential/batch_normalization/batchnorm/mul/ReadVariableOp)^sequential/conv_1/BiasAdd/ReadVariableOp5^sequential/conv_1/Conv1D/ExpandDims_1/ReadVariableOp)^sequential/conv_2/BiasAdd/ReadVariableOp5^sequential/conv_2/Conv1D/ExpandDims_1/ReadVariableOp)^sequential/conv_3/BiasAdd/ReadVariableOp5^sequential/conv_3/Conv1D/ExpandDims_1/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2r
7sequential/batch_normalization/batchnorm/ReadVariableOp7sequential/batch_normalization/batchnorm/ReadVariableOp2v
9sequential/batch_normalization/batchnorm/ReadVariableOp_19sequential/batch_normalization/batchnorm/ReadVariableOp_12v
9sequential/batch_normalization/batchnorm/ReadVariableOp_29sequential/batch_normalization/batchnorm/ReadVariableOp_22z
;sequential/batch_normalization/batchnorm/mul/ReadVariableOp;sequential/batch_normalization/batchnorm/mul/ReadVariableOp2T
(sequential/conv_1/BiasAdd/ReadVariableOp(sequential/conv_1/BiasAdd/ReadVariableOp2l
4sequential/conv_1/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv_1/Conv1D/ExpandDims_1/ReadVariableOp2T
(sequential/conv_2/BiasAdd/ReadVariableOp(sequential/conv_2/BiasAdd/ReadVariableOp2l
4sequential/conv_2/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv_2/Conv1D/ExpandDims_1/ReadVariableOp2T
(sequential/conv_3/BiasAdd/ReadVariableOp(sequential/conv_3/BiasAdd/ReadVariableOp2l
4sequential/conv_3/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv_3/Conv1D/ExpandDims_1/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
(__inference_conv_2_layer_call_fn_3113288

inputs
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
(__inference_conv_3_layer_call_fn_3113336

inputs
unknown:�`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_3113388

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������`*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111596|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������`
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858

inputs2
matmul_readvariableop_resource:
�w�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������w: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������w
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3113493

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����;  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������wY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������w"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
�

G__inference_sequential_layer_call_and_return_conditional_losses_3112628
input_1$
conv_1_3112506:@
conv_1_3112508:@%
conv_2_3112519:@�
conv_2_3112521:	�%
conv_3_3112532:�`
conv_3_3112534:`)
batch_normalization_3112545:`)
batch_normalization_3112547:`)
batch_normalization_3112549:`)
batch_normalization_3112551:`#
dense_1_3112556:
�w�
dense_1_3112558:	�!
output_3112570:	�
output_3112572:
identity

identity_1

identity_2

identity_3

identity_4��+batch_normalization/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp�conv_2/StatefulPartitionedCall�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp�conv_3/StatefulPartitionedCall�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp�dense_1/StatefulPartitionedCall�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�output/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_3111683�
conv_1/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv_1_3112506conv_1_3112508*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713�
*conv_1/ActivityRegularizer/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_1_activity_regularizer_3111546w
 conv_1/ActivityRegularizer/ShapeShape'conv_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_1/ActivityRegularizer/strided_sliceStridedSlice)conv_1/ActivityRegularizer/Shape:output:07conv_1/ActivityRegularizer/strided_slice/stack:output:09conv_1/ActivityRegularizer/strided_slice/stack_1:output:09conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_1/ActivityRegularizer/CastCast1conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_1/ActivityRegularizer/truedivRealDiv3conv_1/ActivityRegularizer/PartitionedCall:output:0#conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_3112519conv_2_3112521*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755�
*conv_2/ActivityRegularizer/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_2_activity_regularizer_3111559w
 conv_2/ActivityRegularizer/ShapeShape'conv_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_2/ActivityRegularizer/strided_sliceStridedSlice)conv_2/ActivityRegularizer/Shape:output:07conv_2/ActivityRegularizer/strided_slice/stack:output:09conv_2/ActivityRegularizer/strided_slice/stack_1:output:09conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_2/ActivityRegularizer/CastCast1conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_2/ActivityRegularizer/truedivRealDiv3conv_2/ActivityRegularizer/PartitionedCall:output:0#conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_3112532conv_3_3112534*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797�
*conv_3/ActivityRegularizer/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_3_activity_regularizer_3111572w
 conv_3/ActivityRegularizer/ShapeShape'conv_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_3/ActivityRegularizer/strided_sliceStridedSlice)conv_3/ActivityRegularizer/Shape:output:07conv_3/ActivityRegularizer/strided_slice/stack:output:09conv_3/ActivityRegularizer/strided_slice/stack_1:output:09conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_3/ActivityRegularizer/CastCast1conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_3/ActivityRegularizer/truedivRealDiv3conv_3/ActivityRegularizer/PartitionedCall:output:0#conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_3112545batch_normalization_3112547batch_normalization_3112549batch_normalization_3112551*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111643�
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3112064�
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������w* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3111833�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_3112556dense_1_3112558*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858�
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *9
f4R2
0__inference_dense_1_activity_regularizer_3111667y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3112013�
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_3112570output_3112572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_3111889
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3112506*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3112508*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3112519*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3112521*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3112532*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3112534*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3112556* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3112558*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������f

Identity_1Identity&conv_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&conv_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_3Identity&conv_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: g

Identity_4Identity'dense_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv_1/StatefulPartitionedCall+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp^conv_2/StatefulPartitionedCall+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp^conv_3/StatefulPartitionedCall+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp ^dense_1/StatefulPartitionedCall,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
�

G__inference_sequential_layer_call_and_return_conditional_losses_3112304

inputs$
conv_1_3112182:@
conv_1_3112184:@%
conv_2_3112195:@�
conv_2_3112197:	�%
conv_3_3112208:�`
conv_3_3112210:`)
batch_normalization_3112221:`)
batch_normalization_3112223:`)
batch_normalization_3112225:`)
batch_normalization_3112227:`#
dense_1_3112232:
�w�
dense_1_3112234:	�!
output_3112246:	�
output_3112248:
identity

identity_1

identity_2

identity_3

identity_4��+batch_normalization/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp�conv_2/StatefulPartitionedCall�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp�conv_3/StatefulPartitionedCall�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp�dense_1/StatefulPartitionedCall�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�output/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_3111683�
conv_1/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv_1_3112182conv_1_3112184*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713�
*conv_1/ActivityRegularizer/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_1_activity_regularizer_3111546w
 conv_1/ActivityRegularizer/ShapeShape'conv_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_1/ActivityRegularizer/strided_sliceStridedSlice)conv_1/ActivityRegularizer/Shape:output:07conv_1/ActivityRegularizer/strided_slice/stack:output:09conv_1/ActivityRegularizer/strided_slice/stack_1:output:09conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_1/ActivityRegularizer/CastCast1conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_1/ActivityRegularizer/truedivRealDiv3conv_1/ActivityRegularizer/PartitionedCall:output:0#conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_3112195conv_2_3112197*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755�
*conv_2/ActivityRegularizer/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_2_activity_regularizer_3111559w
 conv_2/ActivityRegularizer/ShapeShape'conv_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_2/ActivityRegularizer/strided_sliceStridedSlice)conv_2/ActivityRegularizer/Shape:output:07conv_2/ActivityRegularizer/strided_slice/stack:output:09conv_2/ActivityRegularizer/strided_slice/stack_1:output:09conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_2/ActivityRegularizer/CastCast1conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_2/ActivityRegularizer/truedivRealDiv3conv_2/ActivityRegularizer/PartitionedCall:output:0#conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_3112208conv_3_3112210*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797�
*conv_3/ActivityRegularizer/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_3_activity_regularizer_3111572w
 conv_3/ActivityRegularizer/ShapeShape'conv_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_3/ActivityRegularizer/strided_sliceStridedSlice)conv_3/ActivityRegularizer/Shape:output:07conv_3/ActivityRegularizer/strided_slice/stack:output:09conv_3/ActivityRegularizer/strided_slice/stack_1:output:09conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_3/ActivityRegularizer/CastCast1conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_3/ActivityRegularizer/truedivRealDiv3conv_3/ActivityRegularizer/PartitionedCall:output:0#conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_3112221batch_normalization_3112223batch_normalization_3112225batch_normalization_3112227*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111643�
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3112064�
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������w* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3111833�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_3112232dense_1_3112234*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858�
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *9
f4R2
0__inference_dense_1_activity_regularizer_3111667y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3112013�
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_3112246output_3112248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_3111889
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3112182*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3112184*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3112195*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3112197*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3112208*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3112210*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3112232* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3112234*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������f

Identity_1Identity&conv_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&conv_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_3Identity&conv_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: g

Identity_4Identity'dense_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv_1/StatefulPartitionedCall+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp^conv_2/StatefulPartitionedCall+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp^conv_3/StatefulPartitionedCall+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp ^dense_1/StatefulPartitionedCall,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
/__inference_conv_3_activity_regularizer_3111572
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
�
,__inference_sequential_layer_call_fn_3112796

inputs
unknown:@
	unknown_0:@ 
	unknown_1:@�
	unknown_2:	� 
	unknown_3:�`
	unknown_4:`
	unknown_5:`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:
�w�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3111948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
__inference_loss_fn_1_3113604A
3conv_1_bias_regularizer_abs_readvariableop_resource:@
identity��*conv_1/bias/Regularizer/Abs/ReadVariableOp�
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp3conv_1_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: s
NoOpNoOp+^conv_1/bias/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp
�
�
C__inference_conv_2_layer_call_and_return_conditional_losses_3113327

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:������������
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityRelu:activations:0^NoOp*
T0*-
_output_shapes
:������������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
G
+__inference_dropout_1_layer_call_fn_3113541

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3111877a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�V
�
 __inference__traced_save_3113819
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_conv_1_kernel_m_read_readvariableop,
(savev2_conv_1_bias_m_read_readvariableop.
*savev2_conv_2_kernel_m_read_readvariableop,
(savev2_conv_2_bias_m_read_readvariableop.
*savev2_conv_3_kernel_m_read_readvariableop,
(savev2_conv_3_bias_m_read_readvariableop:
6savev2_batch_normalization_gamma_m_read_readvariableop9
5savev2_batch_normalization_beta_m_read_readvariableop/
+savev2_dense_1_kernel_m_read_readvariableop-
)savev2_dense_1_bias_m_read_readvariableop.
*savev2_output_kernel_m_read_readvariableop,
(savev2_output_bias_m_read_readvariableop.
*savev2_conv_1_kernel_v_read_readvariableop,
(savev2_conv_1_bias_v_read_readvariableop.
*savev2_conv_2_kernel_v_read_readvariableop,
(savev2_conv_2_bias_v_read_readvariableop.
*savev2_conv_3_kernel_v_read_readvariableop,
(savev2_conv_3_bias_v_read_readvariableop:
6savev2_batch_normalization_gamma_v_read_readvariableop9
5savev2_batch_normalization_beta_v_read_readvariableop/
+savev2_dense_1_kernel_v_read_readvariableop-
)savev2_dense_1_bias_v_read_readvariableop.
*savev2_output_kernel_v_read_readvariableop,
(savev2_output_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_conv_1_kernel_m_read_readvariableop(savev2_conv_1_bias_m_read_readvariableop*savev2_conv_2_kernel_m_read_readvariableop(savev2_conv_2_bias_m_read_readvariableop*savev2_conv_3_kernel_m_read_readvariableop(savev2_conv_3_bias_m_read_readvariableop6savev2_batch_normalization_gamma_m_read_readvariableop5savev2_batch_normalization_beta_m_read_readvariableop+savev2_dense_1_kernel_m_read_readvariableop)savev2_dense_1_bias_m_read_readvariableop*savev2_output_kernel_m_read_readvariableop(savev2_output_bias_m_read_readvariableop*savev2_conv_1_kernel_v_read_readvariableop(savev2_conv_1_bias_v_read_readvariableop*savev2_conv_2_kernel_v_read_readvariableop(savev2_conv_2_bias_v_read_readvariableop*savev2_conv_3_kernel_v_read_readvariableop(savev2_conv_3_bias_v_read_readvariableop6savev2_batch_normalization_gamma_v_read_readvariableop5savev2_batch_normalization_beta_v_read_readvariableop+savev2_dense_1_kernel_v_read_readvariableop)savev2_dense_1_bias_v_read_readvariableop*savev2_output_kernel_v_read_readvariableop(savev2_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@�:�:�`:`:`:`:`:`:
�w�:�:	�:: : : : :@:@:@�:�:�`:`:`:`:
�w�:�:	�::@:@:@�:�:�`:`:`:`:
�w�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:)%
#
_output_shapes
:�`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 	

_output_shapes
:`: 


_output_shapes
:`:&"
 
_output_shapes
:
�w�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:@: 

_output_shapes
:@:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:)%
#
_output_shapes
:�`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:&"
 
_output_shapes
:
�w�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::($
"
_output_shapes
:@:  

_output_shapes
:@:)!%
#
_output_shapes
:@�:!"

_output_shapes	
:�:)#%
#
_output_shapes
:�`: $

_output_shapes
:`: %

_output_shapes
:`: &

_output_shapes
:`:&'"
 
_output_shapes
:
�w�:!(

_output_shapes	
:�:%)!

_output_shapes
:	�: *

_output_shapes
::+

_output_shapes
: 
�

c
D__inference_dropout_layer_call_and_return_conditional_losses_3112064

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������`C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������`*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������`t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������`n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������`^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
�
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:������������
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityRelu:activations:0^NoOp*
T0*-
_output_shapes
:������������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3111833

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����;  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������wY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������w"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
b
F__inference_rescaling_layer_call_and_return_conditional_losses_3113231

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������Q
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
mulMulinputsCast/x:output:0*
T0*,
_output_shapes
:����������X
addAddV2mul:z:0
Cast_1:y:0*
T0*,
_output_shapes
:����������T
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv_3_layer_call_and_return_conditional_losses_3113375

inputsB
+conv1d_expanddims_1_readvariableop_resource:�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�`�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������`*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������`*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������`U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������`�
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_3113487

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������w* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3111833a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������w"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
d
+__inference_dropout_1_layer_call_fn_3113546

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3112013p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797

inputsB
+conv1d_expanddims_1_readvariableop_resource:�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�`�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������`*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������`*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������`U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������`�
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_3113582

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_dropout_layer_call_fn_3113460

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3111825e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
�
G__inference_conv_3_layer_call_and_return_all_conditional_losses_3113347

inputs
unknown:�`
	unknown_0:`
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797�
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_3_activity_regularizer_3111572t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_3113470

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������``

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
__inference_loss_fn_3_3113626B
3conv_2_bias_regularizer_abs_readvariableop_resource:	�
identity��*conv_2/bias/Regularizer/Abs/ReadVariableOp�
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOp3conv_2_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_2/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: s
NoOpNoOp+^conv_2/bias/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113551

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_3111825

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������``

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
�
G__inference_conv_1_layer_call_and_return_all_conditional_losses_3113251

inputs
unknown:@
	unknown_0:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713�
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_1_activity_regularizer_3111546t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_conv_1_layer_call_fn_3113240

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_3112711
input_1
unknown:@
	unknown_0:@ 
	unknown_1:@�
	unknown_2:	� 
	unknown_3:�`
	unknown_4:`
	unknown_5:`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:
�w�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_3111533o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
(__inference_output_layer_call_fn_3113572

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_3111889o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_3111983
input_1
unknown:@
	unknown_0:@ 
	unknown_1:@�
	unknown_2:	� 
	unknown_3:�`
	unknown_4:`
	unknown_5:`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:
�w�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3111948o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113421

inputs/
!batchnorm_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`1
#batchnorm_readvariableop_1_resource:`1
#batchnorm_readvariableop_2_resource:`
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������`z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:`z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������`o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������`�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������`: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������`
 
_user_specified_nameinputs
�
�
D__inference_dense_1_layer_call_and_return_conditional_losses_3113536

inputs2
matmul_readvariableop_resource:
�w�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������w: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������w
 
_user_specified_nameinputs
�%
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111643

inputs5
'assignmovingavg_readvariableop_resource:`7
)assignmovingavg_1_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`/
!batchnorm_readvariableop_resource:`
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:`*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:`�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������`s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:`*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:`x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:`~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������`h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:`v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������`o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������`�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������`: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������`
 
_user_specified_nameinputs
Ȁ
�
G__inference_sequential_layer_call_and_return_conditional_losses_3113217

inputsH
2conv_1_conv1d_expanddims_1_readvariableop_resource:@4
&conv_1_biasadd_readvariableop_resource:@I
2conv_2_conv1d_expanddims_1_readvariableop_resource:@�5
&conv_2_biasadd_readvariableop_resource:	�I
2conv_3_conv1d_expanddims_1_readvariableop_resource:�`4
&conv_3_biasadd_readvariableop_resource:`I
;batch_normalization_assignmovingavg_readvariableop_resource:`K
=batch_normalization_assignmovingavg_1_readvariableop_resource:`G
9batch_normalization_batchnorm_mul_readvariableop_resource:`C
5batch_normalization_batchnorm_readvariableop_resource:`:
&dense_1_matmul_readvariableop_resource:
�w�6
'dense_1_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�)conv_1/Conv1D/ExpandDims_1/ReadVariableOp�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�)conv_2/Conv1D/ExpandDims_1/ReadVariableOp�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�)conv_3/Conv1D/ExpandDims_1/ReadVariableOp�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <]
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������e
rescaling/Cast_1Castrescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: n
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*,
_output_shapes
:����������v
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1:y:0*
T0*,
_output_shapes
:����������g
conv_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_1/Conv1D/ExpandDims
ExpandDimsrescaling/add:z:0%conv_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
)conv_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0`
conv_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv_1/Conv1D/ExpandDims_1
ExpandDims1conv_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@�
conv_1/Conv1DConv2D!conv_1/Conv1D/ExpandDims:output:0#conv_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
�
conv_1/Conv1D/SqueezeSqueezeconv_1/Conv1D:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

����������
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv_1/BiasAddBiasAddconv_1/Conv1D/Squeeze:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@c
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@w
conv_1/ActivityRegularizer/AbsAbsconv_1/Relu:activations:0*
T0*,
_output_shapes
:����������@u
 conv_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/ActivityRegularizer/SumSum"conv_1/ActivityRegularizer/Abs:y:0)conv_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/ActivityRegularizer/mulMul)conv_1/ActivityRegularizer/mul/x:output:0'conv_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
 conv_1/ActivityRegularizer/ShapeShapeconv_1/Relu:activations:0*
T0*
_output_shapes
:x
.conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_1/ActivityRegularizer/strided_sliceStridedSlice)conv_1/ActivityRegularizer/Shape:output:07conv_1/ActivityRegularizer/strided_slice/stack:output:09conv_1/ActivityRegularizer/strided_slice/stack_1:output:09conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_1/ActivityRegularizer/CastCast1conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_1/ActivityRegularizer/truedivRealDiv"conv_1/ActivityRegularizer/mul:z:0#conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: g
conv_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_2/Conv1D/ExpandDims
ExpandDimsconv_1/Relu:activations:0%conv_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@�
)conv_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0`
conv_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv_2/Conv1D/ExpandDims_1
ExpandDims1conv_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
conv_2/Conv1DConv2D!conv_2/Conv1D/ExpandDims:output:0#conv_2/Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
�
conv_2/Conv1D/SqueezeSqueezeconv_2/Conv1D:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims

����������
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_2/BiasAddBiasAddconv_2/Conv1D/Squeeze:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������d
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*-
_output_shapes
:�����������x
conv_2/ActivityRegularizer/AbsAbsconv_2/Relu:activations:0*
T0*-
_output_shapes
:�����������u
 conv_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/ActivityRegularizer/SumSum"conv_2/ActivityRegularizer/Abs:y:0)conv_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/ActivityRegularizer/mulMul)conv_2/ActivityRegularizer/mul/x:output:0'conv_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
 conv_2/ActivityRegularizer/ShapeShapeconv_2/Relu:activations:0*
T0*
_output_shapes
:x
.conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_2/ActivityRegularizer/strided_sliceStridedSlice)conv_2/ActivityRegularizer/Shape:output:07conv_2/ActivityRegularizer/strided_slice/stack:output:09conv_2/ActivityRegularizer/strided_slice/stack_1:output:09conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_2/ActivityRegularizer/CastCast1conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_2/ActivityRegularizer/truedivRealDiv"conv_2/ActivityRegularizer/mul:z:0#conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: g
conv_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_3/Conv1D/ExpandDims
ExpandDimsconv_2/Relu:activations:0%conv_3/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
)conv_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0`
conv_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv_3/Conv1D/ExpandDims_1
ExpandDims1conv_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�`�
conv_3/Conv1DConv2D!conv_3/Conv1D/ExpandDims:output:0#conv_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������`*
paddingVALID*
strides
�
conv_3/Conv1D/SqueezeSqueezeconv_3/Conv1D:output:0*
T0*,
_output_shapes
:����������`*
squeeze_dims

����������
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
conv_3/BiasAddBiasAddconv_3/Conv1D/Squeeze:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������`c
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������`w
conv_3/ActivityRegularizer/AbsAbsconv_3/Relu:activations:0*
T0*,
_output_shapes
:����������`u
 conv_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/ActivityRegularizer/SumSum"conv_3/ActivityRegularizer/Abs:y:0)conv_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/ActivityRegularizer/mulMul)conv_3/ActivityRegularizer/mul/x:output:0'conv_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
 conv_3/ActivityRegularizer/ShapeShapeconv_3/Relu:activations:0*
T0*
_output_shapes
:x
.conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_3/ActivityRegularizer/strided_sliceStridedSlice)conv_3/ActivityRegularizer/Shape:output:07conv_3/ActivityRegularizer/strided_slice/stack:output:09conv_3/ActivityRegularizer/strided_slice/stack_1:output:09conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_3/ActivityRegularizer/CastCast1conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_3/ActivityRegularizer/truedivRealDiv"conv_3/ActivityRegularizer/mul:z:0#conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
 batch_normalization/moments/meanMeanconv_3/Relu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:`*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:`�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv_3/Relu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������`�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:`*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:`�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:`�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:`x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:`�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`�
#batch_normalization/batchnorm/mul_1Mulconv_3/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������`�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:`�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:`�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������`Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout/dropout/MulMul'batch_normalization/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:����������`l
dropout/dropout/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:����������`*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������`�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������`�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:����������`^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����;  �
flatten/ReshapeReshapedropout/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:����������w�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������u
dense_1/ActivityRegularizer/AbsAbsdense_1/Relu:activations:0*
T0*(
_output_shapes
:����������r
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/ActivityRegularizer/SumSum#dense_1/ActivityRegularizer/Abs:y:0*dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: k
!dense_1/ActivityRegularizer/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_1/ActivityRegularizer/truedivRealDiv#dense_1/ActivityRegularizer/mul:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?�
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������a
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output/MatMulMatMuldropout_1/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������f

Identity_1Identity&conv_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&conv_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_3Identity&conv_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: g

Identity_4Identity'dense_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp*^conv_1/Conv1D/ExpandDims_1/ReadVariableOp+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/Conv1D/ExpandDims_1/ReadVariableOp+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp*^conv_3/Conv1D/ExpandDims_1/ReadVariableOp+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2V
)conv_1/Conv1D/ExpandDims_1/ReadVariableOp)conv_1/Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2V
)conv_2/Conv1D/ExpandDims_1/ReadVariableOp)conv_2/Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2V
)conv_3/Conv1D/ExpandDims_1/ReadVariableOp)conv_3/Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@�
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
__inference_loss_fn_7_3113670C
4dense_1_bias_regularizer_abs_readvariableop_resource:	�
identity��+dense_1/bias/Regularizer/Abs/ReadVariableOp�
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_1_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: t
NoOpNoOp,^dense_1/bias/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp
��
�
G__inference_sequential_layer_call_and_return_conditional_losses_3113011

inputsH
2conv_1_conv1d_expanddims_1_readvariableop_resource:@4
&conv_1_biasadd_readvariableop_resource:@I
2conv_2_conv1d_expanddims_1_readvariableop_resource:@�5
&conv_2_biasadd_readvariableop_resource:	�I
2conv_3_conv1d_expanddims_1_readvariableop_resource:�`4
&conv_3_biasadd_readvariableop_resource:`C
5batch_normalization_batchnorm_readvariableop_resource:`G
9batch_normalization_batchnorm_mul_readvariableop_resource:`E
7batch_normalization_batchnorm_readvariableop_1_resource:`E
7batch_normalization_batchnorm_readvariableop_2_resource:`:
&dense_1_matmul_readvariableop_resource:
�w�6
'dense_1_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�)conv_1/Conv1D/ExpandDims_1/ReadVariableOp�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�)conv_2/Conv1D/ExpandDims_1/ReadVariableOp�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�)conv_3/Conv1D/ExpandDims_1/ReadVariableOp�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�� <]
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������e
rescaling/Cast_1Castrescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: n
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*,
_output_shapes
:����������v
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1:y:0*
T0*,
_output_shapes
:����������g
conv_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_1/Conv1D/ExpandDims
ExpandDimsrescaling/add:z:0%conv_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
)conv_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0`
conv_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv_1/Conv1D/ExpandDims_1
ExpandDims1conv_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@�
conv_1/Conv1DConv2D!conv_1/Conv1D/ExpandDims:output:0#conv_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
�
conv_1/Conv1D/SqueezeSqueezeconv_1/Conv1D:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

����������
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv_1/BiasAddBiasAddconv_1/Conv1D/Squeeze:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@c
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@w
conv_1/ActivityRegularizer/AbsAbsconv_1/Relu:activations:0*
T0*,
_output_shapes
:����������@u
 conv_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/ActivityRegularizer/SumSum"conv_1/ActivityRegularizer/Abs:y:0)conv_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/ActivityRegularizer/mulMul)conv_1/ActivityRegularizer/mul/x:output:0'conv_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
 conv_1/ActivityRegularizer/ShapeShapeconv_1/Relu:activations:0*
T0*
_output_shapes
:x
.conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_1/ActivityRegularizer/strided_sliceStridedSlice)conv_1/ActivityRegularizer/Shape:output:07conv_1/ActivityRegularizer/strided_slice/stack:output:09conv_1/ActivityRegularizer/strided_slice/stack_1:output:09conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_1/ActivityRegularizer/CastCast1conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_1/ActivityRegularizer/truedivRealDiv"conv_1/ActivityRegularizer/mul:z:0#conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: g
conv_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_2/Conv1D/ExpandDims
ExpandDimsconv_1/Relu:activations:0%conv_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@�
)conv_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0`
conv_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv_2/Conv1D/ExpandDims_1
ExpandDims1conv_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv_2/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
conv_2/Conv1DConv2D!conv_2/Conv1D/ExpandDims:output:0#conv_2/Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
�
conv_2/Conv1D/SqueezeSqueezeconv_2/Conv1D:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims

����������
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_2/BiasAddBiasAddconv_2/Conv1D/Squeeze:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������d
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*-
_output_shapes
:�����������x
conv_2/ActivityRegularizer/AbsAbsconv_2/Relu:activations:0*
T0*-
_output_shapes
:�����������u
 conv_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/ActivityRegularizer/SumSum"conv_2/ActivityRegularizer/Abs:y:0)conv_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/ActivityRegularizer/mulMul)conv_2/ActivityRegularizer/mul/x:output:0'conv_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
 conv_2/ActivityRegularizer/ShapeShapeconv_2/Relu:activations:0*
T0*
_output_shapes
:x
.conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_2/ActivityRegularizer/strided_sliceStridedSlice)conv_2/ActivityRegularizer/Shape:output:07conv_2/ActivityRegularizer/strided_slice/stack:output:09conv_2/ActivityRegularizer/strided_slice/stack_1:output:09conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_2/ActivityRegularizer/CastCast1conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_2/ActivityRegularizer/truedivRealDiv"conv_2/ActivityRegularizer/mul:z:0#conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: g
conv_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_3/Conv1D/ExpandDims
ExpandDimsconv_2/Relu:activations:0%conv_3/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
)conv_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0`
conv_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv_3/Conv1D/ExpandDims_1
ExpandDims1conv_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:�`�
conv_3/Conv1DConv2D!conv_3/Conv1D/ExpandDims:output:0#conv_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������`*
paddingVALID*
strides
�
conv_3/Conv1D/SqueezeSqueezeconv_3/Conv1D:output:0*
T0*,
_output_shapes
:����������`*
squeeze_dims

����������
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
conv_3/BiasAddBiasAddconv_3/Conv1D/Squeeze:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������`c
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������`w
conv_3/ActivityRegularizer/AbsAbsconv_3/Relu:activations:0*
T0*,
_output_shapes
:����������`u
 conv_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/ActivityRegularizer/SumSum"conv_3/ActivityRegularizer/Abs:y:0)conv_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/ActivityRegularizer/mulMul)conv_3/ActivityRegularizer/mul/x:output:0'conv_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: i
 conv_3/ActivityRegularizer/ShapeShapeconv_3/Relu:activations:0*
T0*
_output_shapes
:x
.conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_3/ActivityRegularizer/strided_sliceStridedSlice)conv_3/ActivityRegularizer/Shape:output:07conv_3/ActivityRegularizer/strided_slice/stack:output:09conv_3/ActivityRegularizer/strided_slice/stack_1:output:09conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_3/ActivityRegularizer/CastCast1conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_3/ActivityRegularizer/truedivRealDiv"conv_3/ActivityRegularizer/mul:z:0#conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:`x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:`�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`�
#batch_normalization/batchnorm/mul_1Mulconv_3/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������`�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:`�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:`�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������`|
dropout/IdentityIdentity'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:����������`^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����;  �
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������w�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������u
dense_1/ActivityRegularizer/AbsAbsdense_1/Relu:activations:0*
T0*(
_output_shapes
:����������r
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/ActivityRegularizer/SumSum#dense_1/ActivityRegularizer/Abs:y:0*dense_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: k
!dense_1/ActivityRegularizer/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_1/ActivityRegularizer/truedivRealDiv#dense_1/ActivityRegularizer/mul:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: m
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*(
_output_shapes
:�����������
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output/MatMulMatMuldropout_1/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������f

Identity_1Identity&conv_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&conv_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_3Identity&conv_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: g

Identity_4Identity'dense_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp*^conv_1/Conv1D/ExpandDims_1/ReadVariableOp+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/Conv1D/ExpandDims_1/ReadVariableOp+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp*^conv_3/Conv1D/ExpandDims_1/ReadVariableOp+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2V
)conv_1/Conv1D/ExpandDims_1/ReadVariableOp)conv_1/Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2V
)conv_2/Conv1D/ExpandDims_1/ReadVariableOp)conv_2/Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2V
)conv_3/Conv1D/ExpandDims_1/ReadVariableOp)conv_3/Conv1D/ExpandDims_1/ReadVariableOp2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_3112833

inputs
unknown:@
	unknown_0:@ 
	unknown_1:@�
	unknown_2:	� 
	unknown_3:�`
	unknown_4:`
	unknown_5:`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:
�w�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:���������: : : : *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_3112304o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
/__inference_conv_1_activity_regularizer_3111546
x
identity0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�
�
5__inference_batch_normalization_layer_call_fn_3113401

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111643|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������`
 
_user_specified_nameinputs
��
�	
G__inference_sequential_layer_call_and_return_conditional_losses_3111948

inputs$
conv_1_3111714:@
conv_1_3111716:@%
conv_2_3111756:@�
conv_2_3111758:	�%
conv_3_3111798:�`
conv_3_3111800:`)
batch_normalization_3111811:`)
batch_normalization_3111813:`)
batch_normalization_3111815:`)
batch_normalization_3111817:`#
dense_1_3111859:
�w�
dense_1_3111861:	�!
output_3111890:	�
output_3111892:
identity

identity_1

identity_2

identity_3

identity_4��+batch_normalization/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�*conv_1/bias/Regularizer/Abs/ReadVariableOp�,conv_1/kernel/Regularizer/Abs/ReadVariableOp�conv_2/StatefulPartitionedCall�*conv_2/bias/Regularizer/Abs/ReadVariableOp�,conv_2/kernel/Regularizer/Abs/ReadVariableOp�conv_3/StatefulPartitionedCall�*conv_3/bias/Regularizer/Abs/ReadVariableOp�,conv_3/kernel/Regularizer/Abs/ReadVariableOp�dense_1/StatefulPartitionedCall�+dense_1/bias/Regularizer/Abs/ReadVariableOp�-dense_1/kernel/Regularizer/Abs/ReadVariableOp�output/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_3111683�
conv_1/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv_1_3111714conv_1_3111716*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_3111713�
*conv_1/ActivityRegularizer/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_1_activity_regularizer_3111546w
 conv_1/ActivityRegularizer/ShapeShape'conv_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_1/ActivityRegularizer/strided_sliceStridedSlice)conv_1/ActivityRegularizer/Shape:output:07conv_1/ActivityRegularizer/strided_slice/stack:output:09conv_1/ActivityRegularizer/strided_slice/stack_1:output:09conv_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_1/ActivityRegularizer/CastCast1conv_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_1/ActivityRegularizer/truedivRealDiv3conv_1/ActivityRegularizer/PartitionedCall:output:0#conv_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_3111756conv_2_3111758*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_3111755�
*conv_2/ActivityRegularizer/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_2_activity_regularizer_3111559w
 conv_2/ActivityRegularizer/ShapeShape'conv_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_2/ActivityRegularizer/strided_sliceStridedSlice)conv_2/ActivityRegularizer/Shape:output:07conv_2/ActivityRegularizer/strided_slice/stack:output:09conv_2/ActivityRegularizer/strided_slice/stack_1:output:09conv_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_2/ActivityRegularizer/CastCast1conv_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_2/ActivityRegularizer/truedivRealDiv3conv_2/ActivityRegularizer/PartitionedCall:output:0#conv_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_3111798conv_3_3111800*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_3111797�
*conv_3/ActivityRegularizer/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *8
f3R1
/__inference_conv_3_activity_regularizer_3111572w
 conv_3/ActivityRegularizer/ShapeShape'conv_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(conv_3/ActivityRegularizer/strided_sliceStridedSlice)conv_3/ActivityRegularizer/Shape:output:07conv_3/ActivityRegularizer/strided_slice/stack:output:09conv_3/ActivityRegularizer/strided_slice/stack_1:output:09conv_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv_3/ActivityRegularizer/CastCast1conv_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"conv_3/ActivityRegularizer/truedivRealDiv3conv_3/ActivityRegularizer/PartitionedCall:output:0#conv_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_3111811batch_normalization_3111813batch_normalization_3111815batch_normalization_3111817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3111596�
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3111825�
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������w* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3111833�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_3111859dense_1_3111861*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3111858�
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
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
GPU 2J 8� *9
f4R2
0__inference_dense_1_activity_regularizer_3111667y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3111877�
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_3111890output_3111892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_3111889
,conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3111714*"
_output_shapes
:@*
dtype0�
conv_1/kernel/Regularizer/AbsAbs4conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@t
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_1/kernel/Regularizer/SumSum!conv_1/kernel/Regularizer/Abs:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_1_3111716*
_output_shapes
:@*
dtype0{
conv_1/bias/Regularizer/AbsAbs2conv_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:@g
conv_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_1/bias/Regularizer/SumSumconv_1/bias/Regularizer/Abs:y:0&conv_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_1/bias/Regularizer/mulMul&conv_1/bias/Regularizer/mul/x:output:0$conv_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3111756*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
*conv_2/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_2_3111758*
_output_shapes	
:�*
dtype0|
conv_2/bias/Regularizer/AbsAbs2conv_2/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�g
conv_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_2/bias/Regularizer/SumSumconv_2/bias/Regularizer/Abs:y:0&conv_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_2/bias/Regularizer/mulMul&conv_2/bias/Regularizer/mul/x:output:0$conv_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
,conv_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3111798*#
_output_shapes
:�`*
dtype0�
conv_3/kernel/Regularizer/AbsAbs4conv_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:�`t
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_3/kernel/Regularizer/SumSum!conv_3/kernel/Regularizer/Abs:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOpconv_3_3111800*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3111859* 
_output_shapes
:
�w�*
dtype0�
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�w�q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
+dense_1/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3111861*
_output_shapes	
:�*
dtype0~
dense_1/bias/Regularizer/AbsAbs3dense_1/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/bias/Regularizer/SumSum dense_1/bias/Regularizer/Abs:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������f

Identity_1Identity&conv_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&conv_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: f

Identity_3Identity&conv_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: g

Identity_4Identity'dense_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv_1/StatefulPartitionedCall+^conv_1/bias/Regularizer/Abs/ReadVariableOp-^conv_1/kernel/Regularizer/Abs/ReadVariableOp^conv_2/StatefulPartitionedCall+^conv_2/bias/Regularizer/Abs/ReadVariableOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp^conv_3/StatefulPartitionedCall+^conv_3/bias/Regularizer/Abs/ReadVariableOp-^conv_3/kernel/Regularizer/Abs/ReadVariableOp ^dense_1/StatefulPartitionedCall,^dense_1/bias/Regularizer/Abs/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2X
*conv_1/bias/Regularizer/Abs/ReadVariableOp*conv_1/bias/Regularizer/Abs/ReadVariableOp2\
,conv_1/kernel/Regularizer/Abs/ReadVariableOp,conv_1/kernel/Regularizer/Abs/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2X
*conv_2/bias/Regularizer/Abs/ReadVariableOp*conv_2/bias/Regularizer/Abs/ReadVariableOp2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp2\
,conv_3/kernel/Regularizer/Abs/ReadVariableOp,conv_3/kernel/Regularizer/Abs/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+dense_1/bias/Regularizer/Abs/ReadVariableOp+dense_1/bias/Regularizer/Abs/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
__inference_loss_fn_5_3113648A
3conv_3_bias_regularizer_abs_readvariableop_resource:`
identity��*conv_3/bias/Regularizer/Abs/ReadVariableOp�
*conv_3/bias/Regularizer/Abs/ReadVariableOpReadVariableOp3conv_3_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:`*
dtype0{
conv_3/bias/Regularizer/AbsAbs2conv_3/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:`g
conv_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
conv_3/bias/Regularizer/SumSumconv_3/bias/Regularizer/Abs:y:0&conv_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
conv_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * �C=�
conv_3/bias/Regularizer/mulMul&conv_3/bias/Regularizer/mul/x:output:0$conv_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityconv_3/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: s
NoOpNoOp+^conv_3/bias/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2X
*conv_3/bias/Regularizer/Abs/ReadVariableOp*conv_3/bias/Regularizer/Abs/ReadVariableOp
�
�
__inference_loss_fn_2_3113615L
5conv_2_kernel_regularizer_abs_readvariableop_resource:@�
identity��,conv_2/kernel/Regularizer/Abs/ReadVariableOp�
,conv_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5conv_2_kernel_regularizer_abs_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
conv_2/kernel/Regularizer/AbsAbs4conv_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�t
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          �
conv_2/kernel/Regularizer/SumSum!conv_2/kernel/Regularizer/Abs:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *)��7�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^conv_2/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,conv_2/kernel/Regularizer/Abs/ReadVariableOp,conv_2/kernel/Regularizer/Abs/ReadVariableOp
�	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_3112013

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input_15
serving_default_input_1:0����������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias
 "_jit_compiled_convolution_op"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias
 +_jit_compiled_convolution_op"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_random_generator"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[_random_generator"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias"
_tf_keras_layer
�
 0
!1
)2
*3
24
35
<6
=7
>8
?9
S10
T11
b12
c13"
trackable_list_wrapper
v
 0
!1
)2
*3
24
35
<6
=7
S8
T9
b10
c11"
trackable_list_wrapper
X
d0
e1
f2
g3
h4
i5
j6
k7"
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
qtrace_0
rtrace_1
strace_2
ttrace_32�
,__inference_sequential_layer_call_fn_3111983
,__inference_sequential_layer_call_fn_3112796
,__inference_sequential_layer_call_fn_3112833
,__inference_sequential_layer_call_fn_3112376�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0zrtrace_1zstrace_2zttrace_3
�
utrace_0
vtrace_1
wtrace_2
xtrace_32�
G__inference_sequential_layer_call_and_return_conditional_losses_3113011
G__inference_sequential_layer_call_and_return_conditional_losses_3113217
G__inference_sequential_layer_call_and_return_conditional_losses_3112502
G__inference_sequential_layer_call_and_return_conditional_losses_3112628�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0zvtrace_1zwtrace_2zxtrace_3
�B�
"__inference__wrapped_model_3111533input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
� m�!m�)m�*m�2m�3m�<m�=m�Sm�Tm�bm�cm� v�!v�)v�*v�2v�3v�<v�=v�Sv�Tv�bv�cv�"
	optimizer
,
yserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
+__inference_rescaling_layer_call_fn_3113222�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
�trace_02�
F__inference_rescaling_layer_call_and_return_conditional_losses_3113231�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
�activity_regularizer_fn
*&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv_1_layer_call_fn_3113240�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_1_layer_call_and_return_all_conditional_losses_3113251�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!@2conv_1/kernel
:@2conv_1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
�activity_regularizer_fn
*(&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv_2_layer_call_fn_3113288�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_2_layer_call_and_return_all_conditional_losses_3113299�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"@�2conv_2/kernel
:�2conv_2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
�activity_regularizer_fn
*1&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv_3_layer_call_fn_3113336�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_3_layer_call_and_return_all_conditional_losses_3113347�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"�`2conv_3/kernel
:`2conv_3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_layer_call_fn_3113388
5__inference_batch_normalization_layer_call_fn_3113401�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113421
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113455�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
':%`2batch_normalization/gamma
&:$`2batch_normalization/beta
/:-` (2batch_normalization/moving_mean
3:1` (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_layer_call_fn_3113460
)__inference_dropout_layer_call_fn_3113465�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_layer_call_and_return_conditional_losses_3113470
D__inference_dropout_layer_call_and_return_conditional_losses_3113482�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_layer_call_fn_3113487�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_layer_call_and_return_conditional_losses_3113493�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
�activity_regularizer_fn
*R&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_1_layer_call_fn_3113502�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_dense_1_layer_call_and_return_all_conditional_losses_3113513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
�w�2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_1_layer_call_fn_3113541
+__inference_dropout_1_layer_call_fn_3113546�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113551
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113563�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_output_layer_call_fn_3113572�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_output_layer_call_and_return_conditional_losses_3113582�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :	�2output/kernel
:2output/bias
�
�trace_02�
__inference_loss_fn_0_3113593�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_3113604�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_3113615�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_3113626�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_3113637�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_3113648�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_3113659�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_3113670�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
.
>0
?1"
trackable_list_wrapper
f
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
9"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_layer_call_fn_3111983input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_3112796inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_3112833inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_3112376input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_3113011inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_3113217inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_3112502input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_3112628input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_3112711input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_rescaling_layer_call_fn_3113222inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_rescaling_layer_call_and_return_conditional_losses_3113231inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�trace_02�
/__inference_conv_1_activity_regularizer_3111546�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�z�trace_0
�
�trace_02�
C__inference_conv_1_layer_call_and_return_conditional_losses_3113279�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
(__inference_conv_1_layer_call_fn_3113240inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_1_layer_call_and_return_all_conditional_losses_3113251inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�trace_02�
/__inference_conv_2_activity_regularizer_3111559�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�z�trace_0
�
�trace_02�
C__inference_conv_2_layer_call_and_return_conditional_losses_3113327�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
(__inference_conv_2_layer_call_fn_3113288inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_2_layer_call_and_return_all_conditional_losses_3113299inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�trace_02�
/__inference_conv_3_activity_regularizer_3111572�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�z�trace_0
�
�trace_02�
C__inference_conv_3_layer_call_and_return_conditional_losses_3113375�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
(__inference_conv_3_layer_call_fn_3113336inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_3_layer_call_and_return_all_conditional_losses_3113347inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_layer_call_fn_3113388inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_layer_call_fn_3113401inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113421inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113455inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_dropout_layer_call_fn_3113460inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_layer_call_fn_3113465inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_3113470inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_3113482inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_flatten_layer_call_fn_3113487inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_layer_call_and_return_conditional_losses_3113493inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�trace_02�
0__inference_dense_1_activity_regularizer_3111667�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�z�trace_0
�
�trace_02�
D__inference_dense_1_layer_call_and_return_conditional_losses_3113536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�B�
)__inference_dense_1_layer_call_fn_3113502inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_1_layer_call_and_return_all_conditional_losses_3113513inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_dropout_1_layer_call_fn_3113541inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_1_layer_call_fn_3113546inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113551inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113563inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_output_layer_call_fn_3113572inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_output_layer_call_and_return_conditional_losses_3113582inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_3113593"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_3113604"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_3113615"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_3113626"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_3113637"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_3113648"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_3113659"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_7_3113670"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
�B�
/__inference_conv_1_activity_regularizer_3111546x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�B�
C__inference_conv_1_layer_call_and_return_conditional_losses_3113279inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_conv_2_activity_regularizer_3111559x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�B�
C__inference_conv_2_layer_call_and_return_conditional_losses_3113327inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_conv_3_activity_regularizer_3111572x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�B�
C__inference_conv_3_layer_call_and_return_conditional_losses_3113375inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_dense_1_activity_regularizer_3111667x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�B�
D__inference_dense_1_layer_call_and_return_conditional_losses_3113536inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
#:!@2conv_1/kernel/m
:@2conv_1/bias/m
$:"@�2conv_2/kernel/m
:�2conv_2/bias/m
$:"�`2conv_3/kernel/m
:`2conv_3/bias/m
':%`2batch_normalization/gamma/m
&:$`2batch_normalization/beta/m
": 
�w�2dense_1/kernel/m
:�2dense_1/bias/m
 :	�2output/kernel/m
:2output/bias/m
#:!@2conv_1/kernel/v
:@2conv_1/bias/v
$:"@�2conv_2/kernel/v
:�2conv_2/bias/v
$:"�`2conv_3/kernel/v
:`2conv_3/bias/v
':%`2batch_normalization/gamma/v
&:$`2batch_normalization/beta/v
": 
�w�2dense_1/kernel/v
:�2dense_1/bias/v
 :	�2output/kernel/v
:2output/bias/v�
"__inference__wrapped_model_3111533x !)*23?<>=STbc5�2
+�(
&�#
input_1����������
� "/�,
*
output �
output����������
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113421|?<>=@�=
6�3
-�*
inputs������������������`
p 
� "2�/
(�%
0������������������`
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3113455|>?<=@�=
6�3
-�*
inputs������������������`
p
� "2�/
(�%
0������������������`
� �
5__inference_batch_normalization_layer_call_fn_3113388o?<>=@�=
6�3
-�*
inputs������������������`
p 
� "%�"������������������`�
5__inference_batch_normalization_layer_call_fn_3113401o>?<=@�=
6�3
-�*
inputs������������������`
p
� "%�"������������������`Y
/__inference_conv_1_activity_regularizer_3111546&�
�
�	
x
� "� �
G__inference_conv_1_layer_call_and_return_all_conditional_losses_3113251t !4�1
*�'
%�"
inputs����������
� "8�5
 �
0����������@
�
�	
1/0 �
C__inference_conv_1_layer_call_and_return_conditional_losses_3113279f !4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
(__inference_conv_1_layer_call_fn_3113240Y !4�1
*�'
%�"
inputs����������
� "�����������@Y
/__inference_conv_2_activity_regularizer_3111559&�
�
�	
x
� "� �
G__inference_conv_2_layer_call_and_return_all_conditional_losses_3113299u)*4�1
*�'
%�"
inputs����������@
� "9�6
!�
0�����������
�
�	
1/0 �
C__inference_conv_2_layer_call_and_return_conditional_losses_3113327g)*4�1
*�'
%�"
inputs����������@
� "+�(
!�
0�����������
� �
(__inference_conv_2_layer_call_fn_3113288Z)*4�1
*�'
%�"
inputs����������@
� "������������Y
/__inference_conv_3_activity_regularizer_3111572&�
�
�	
x
� "� �
G__inference_conv_3_layer_call_and_return_all_conditional_losses_3113347u235�2
+�(
&�#
inputs�����������
� "8�5
 �
0����������`
�
�	
1/0 �
C__inference_conv_3_layer_call_and_return_conditional_losses_3113375g235�2
+�(
&�#
inputs�����������
� "*�'
 �
0����������`
� �
(__inference_conv_3_layer_call_fn_3113336Z235�2
+�(
&�#
inputs�����������
� "�����������`Z
0__inference_dense_1_activity_regularizer_3111667&�
�
�	
x
� "� �
H__inference_dense_1_layer_call_and_return_all_conditional_losses_3113513lST0�-
&�#
!�
inputs����������w
� "4�1
�
0����������
�
�	
1/0 �
D__inference_dense_1_layer_call_and_return_conditional_losses_3113536^ST0�-
&�#
!�
inputs����������w
� "&�#
�
0����������
� ~
)__inference_dense_1_layer_call_fn_3113502QST0�-
&�#
!�
inputs����������w
� "������������
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113551^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_dropout_1_layer_call_and_return_conditional_losses_3113563^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
+__inference_dropout_1_layer_call_fn_3113541Q4�1
*�'
!�
inputs����������
p 
� "������������
+__inference_dropout_1_layer_call_fn_3113546Q4�1
*�'
!�
inputs����������
p
� "������������
D__inference_dropout_layer_call_and_return_conditional_losses_3113470f8�5
.�+
%�"
inputs����������`
p 
� "*�'
 �
0����������`
� �
D__inference_dropout_layer_call_and_return_conditional_losses_3113482f8�5
.�+
%�"
inputs����������`
p
� "*�'
 �
0����������`
� �
)__inference_dropout_layer_call_fn_3113460Y8�5
.�+
%�"
inputs����������`
p 
� "�����������`�
)__inference_dropout_layer_call_fn_3113465Y8�5
.�+
%�"
inputs����������`
p
� "�����������`�
D__inference_flatten_layer_call_and_return_conditional_losses_3113493^4�1
*�'
%�"
inputs����������`
� "&�#
�
0����������w
� ~
)__inference_flatten_layer_call_fn_3113487Q4�1
*�'
%�"
inputs����������`
� "�����������w<
__inference_loss_fn_0_3113593 �

� 
� "� <
__inference_loss_fn_1_3113604!�

� 
� "� <
__inference_loss_fn_2_3113615)�

� 
� "� <
__inference_loss_fn_3_3113626*�

� 
� "� <
__inference_loss_fn_4_31136372�

� 
� "� <
__inference_loss_fn_5_31136483�

� 
� "� <
__inference_loss_fn_6_3113659S�

� 
� "� <
__inference_loss_fn_7_3113670T�

� 
� "� �
C__inference_output_layer_call_and_return_conditional_losses_3113582]bc0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_output_layer_call_fn_3113572Pbc0�-
&�#
!�
inputs����������
� "�����������
F__inference_rescaling_layer_call_and_return_conditional_losses_3113231b4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
+__inference_rescaling_layer_call_fn_3113222U4�1
*�'
%�"
inputs����������
� "������������
G__inference_sequential_layer_call_and_return_conditional_losses_3112502� !)*23?<>=STbc=�:
3�0
&�#
input_1����������
p 

 
� "]�Z
�
0���������
;�8
�	
1/0 
�	
1/1 
�	
1/2 
�	
1/3 �
G__inference_sequential_layer_call_and_return_conditional_losses_3112628� !)*23>?<=STbc=�:
3�0
&�#
input_1����������
p

 
� "]�Z
�
0���������
;�8
�	
1/0 
�	
1/1 
�	
1/2 
�	
1/3 �
G__inference_sequential_layer_call_and_return_conditional_losses_3113011� !)*23?<>=STbc<�9
2�/
%�"
inputs����������
p 

 
� "]�Z
�
0���������
;�8
�	
1/0 
�	
1/1 
�	
1/2 
�	
1/3 �
G__inference_sequential_layer_call_and_return_conditional_losses_3113217� !)*23>?<=STbc<�9
2�/
%�"
inputs����������
p

 
� "]�Z
�
0���������
;�8
�	
1/0 
�	
1/1 
�	
1/2 
�	
1/3 �
,__inference_sequential_layer_call_fn_3111983i !)*23?<>=STbc=�:
3�0
&�#
input_1����������
p 

 
� "�����������
,__inference_sequential_layer_call_fn_3112376i !)*23>?<=STbc=�:
3�0
&�#
input_1����������
p

 
� "�����������
,__inference_sequential_layer_call_fn_3112796h !)*23?<>=STbc<�9
2�/
%�"
inputs����������
p 

 
� "�����������
,__inference_sequential_layer_call_fn_3112833h !)*23>?<=STbc<�9
2�/
%�"
inputs����������
p

 
� "�����������
%__inference_signature_wrapper_3112711� !)*23?<>=STbc@�=
� 
6�3
1
input_1&�#
input_1����������"/�,
*
output �
output���������