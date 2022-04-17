import string

def generate_outer_product_equation(dim1, dim2):

    first_operand_dim = string.ascii_lowercase[0:dim1]
    second_operand_dim = string.ascii_lowercase[dim1:dim1+dim2]

    return f'{first_operand_dim},{second_operand_dim}->{first_operand_dim}{second_operand_dim}'