
# IMPLEMENTING A MATRIX MULTIPLICATION FUNCTION FROM SCRATCH
# THIS FUNCTION IS NOT EXACTLY THE MOST EFFICIENT ALGORITHM TO WORK WITH
# BUT I AM AVOIDING RELYING SOLELY ON NUMPY SYNTAX TO DO MY DIRTY WORK FOR ME
# THIS IS THE SLIGHLTY IMPROVED VERSION WITH VALIDATION AND POLISHING
# FINAL VERSION WITH FIXED VALIDATOR FUNCTION


# changed name to dot product as it is more mathematically accurate
def dot_product(x, y): # both are lists but x is a row and y is column
                              # of the respective matrices they come from
  total = 0
  for i in range(len(x)):
    total = total+ (x[i]*y[i])
  return total # returns a

# im using the function below to index a column of a matrix since i am not too familiar with comprehensions
# to be or not to be, that is the question. 
def col(b, col): # converts a column of a 2-d list/list-of-lists to a list
                 # 'col' is the index of the column we want to convert
  col_list =list()
  for i in range(len(b)):
      col_list.append(b[i][col]) # append the index of row i and column 'col' to list

  return col_list



# def init_matrix(len_a): # takes number of rows of matrix a as input
#                         # returns an empty matrix with len(a) rows
#   mat = list()

#   for i in range(len_a):
#     mat.append([]) # rows of resultant matrix must equal rows of a matrix
#                    # the columns of result will be handled automatically through
#                    # the "multiply->add->append" logic of the nested loop in the
#                    # mat_mult() function
#   return mat
def init_matrix(rows, cols): # takes number of rows and columns as input
                             # returns a zero matrix with same number of rows and columns
  mat = list()

  for i in range(rows):
    mat.append([]) # create a new row i
    for j in range(cols): 
      mat[i].append(0) # create new column by appending 0 to the list in row i. do this j times and you got j columns
  return mat          # [0, 0, 0,....],
                      # [0, 0, 0,....],
                      # [0, 0, 0,....],

# chatGPT generated fixed validator
# im sorry god. i have failed you
def validate_matrices(a, b):

  # Matrix exists
  if len(a) == 0:
    print("Matrix a is empty")
    return False

  if len(b) == 0:
    print("Matrix b is empty")
    return False

  # First row exists and isn't empty
  if len(a[0]) == 0:
    print("Empty Row in a")
    return False

  if len(b[0]) == 0:
    print("Empty Row in b")
    return False

  length_of_row_a = len(a[0])
  length_of_row_b = len(b[0])

  # Check every row in A
  for i in range(len(a)):
    if len(a[i]) == 0:
      print("Empty Row in a")
      return False

    if len(a[i]) != length_of_row_a:
      print("Malformed Matrix a")
      return False

  # Check every row in B
  for i in range(len(b)):
    if len(b[i]) == 0:
      print("Empty Row in b")
      return False

    if len(b[i]) != length_of_row_b:
      print("Malformed Matrix b")
      return False

  # Finally, check multiplication compatibility
  if length_of_row_a != len(b):
    print("Dimension mismatch. This is not a valid matrix multiplication")
    return False

  return True



def mat_mult(a, b): # 'a' and 'b' are both matrices.
                    #  in this case we will represent matrices as a list-of-lists
# len(x) => number of rows
# len(x[0]) => number of columns
  if validate_matrices(a, b):
    result = init_matrix(len(a), len(b[0])) # 'result' could be another matrix (list-of-lists) or just a vector (list) depending on the input dimensions
    for i in range(len(a)):
      for j in range(len(b[0])):
        # append this list to row i of the resultant list-of-lists
        result[i][j]=(dot_product(a[i], col(b, j)))
    return result

## ADDITIONAL FUNCTION FOR MATRIX ADDITION
def mat_add(a, b):
  # len(x) => number of rows
# len(x[0]) => number of columns
  if len(a)  == len(b) and len(a[0]) == len(b[0]):
    result = init_matrix(len(a), len(a[0]))
    for i in range(len(a)):
      for j in range(len(a[0])):
        result[i][j]=(a[i][j]+b[i][j])
  return result


if __name__=='__main__':
  a = [[1, 0],
       [0, 1]]
  b = [[0, 1, 2],
       [1, -1, 3]]
  print(mat_mult(a, b))

  a = [[1, 0], [0, 1]]
  b = [[1, 0], [0, 1]]
  print(mat_add(a, b))
