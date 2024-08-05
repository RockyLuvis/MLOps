import sys

class CustomException(Exception):

    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.line_num = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename


    def __str__(self):
        try:
            return "Error occured in python script name [{0}] line number [{1}] with error message [{2}]". format(self.file_name, self.line_num, self.error_message)
        except Exception as e:
            return f"An error occurred while formatting the error message: {str(e)}"



if __name__=="__main__":
    try:
        a=10/0

    except Exception as e:
        #print (e)
        raise CustomException(str(e), sys)