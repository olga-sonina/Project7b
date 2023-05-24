import streamlit as st

st.write("# Home credit App")


st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects."""

    )


# def mapping_demo():
# 	st.write('the second page')

# def plotting_demo():
# 	st.write('the third page')

# def data_frame_demo():
# 	st.write('the forth page')

# page_names_to_funcs = {
#     "—": intro,
#     "Plotting Demo": plotting_demo,
#     "Mapping Demo": mapping_demo,
#     "DataFrame Demo": data_frame_demo
# }

# demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
# page_names_to_funcs[demo_name]()