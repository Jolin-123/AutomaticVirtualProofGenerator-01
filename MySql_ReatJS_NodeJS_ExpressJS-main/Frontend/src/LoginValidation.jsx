function validation(values){

    let error ={}
    const email_pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    //const password_pattern = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    const password_pattern = /^.{4,}$/;


    if (values.email === "") {
        error.email = "Email should not be empty"
    }

    else if (!email_pattern.test(values.email)){
        error.email = "Email Dees not match "
    } else {
        error.email = ""
    }


    if (values.password === "") {
        error.password = "Password should not be empty"
    }

    else if (!password_pattern.test(values.password)){
        error.password = "Password does not match "
    } else {
        error.password = ""
    }

    return error; 

}

export default validation;