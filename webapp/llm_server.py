from ollama import chat
from ollama import ChatResponse

def advisor(input_df):
    pre_prompt = """
    You are the Director of customer relations at a telephone company competing with T-Mobile and Google Fi. Your job is to analyze ways to retain customers from churning. You cannot employ technical methods for this. You have data that tells you the services that customers have opted for, what their demographic is and what kind of contracts they are on with the company. YOu also have information if they have resigned or not, which essentially means you know everything about a customer and if they are churning or not. In the given dataset, if churn is True, it means that the customer has churned else, if the Churn is false, you have retained the customer. You are given the churn value of a certain customer with the following features: """

    in_dat = ""
    for i in input_df.columns:
        print(input_df[i][0])
        in_dat=in_dat+str(i)+": "+str(input_df[i][0])+"\n"
    input = in_dat

    post_prompt_true = """How will you prevent them from churning, without approaching them directly about this? How would you retain other several customers with similar features?\n"""

    post_prompt_false = """So far your company has managed to retain customers of this kind successfully. How will you improve your services such that the churn probability remains low like in this case?"""

    rules = """Rules: Follow these rules in the order of their priority:\n1)Keep your response short and to the point\n2)Use important keywords and bullet points.\n3)Do not ask follow up questions.\n4)Do not include your thinking process.\n5)Only and only include instructions for the marketting personnel and other employees in your company to follow."""

    if input_df['Churn'][0]:
        prompt = pre_prompt+"\n"+input+"\n"+post_prompt_true+rules
    elif input_df['Churn'][0] == False:
        prompt = pre_prompt+"\n"+input+"\n"+post_prompt_false+rules
    print(prompt)

    response: ChatResponse = chat(model='gemma3:4b', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    print(response['message']['content'])
    print(response.message.content)
    return(response['message']['content'])
