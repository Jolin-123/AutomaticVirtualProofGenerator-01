import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import validation from './LoginValidation';
import './signinupStyle.css';
//import 'boostrap/dist/css/boostrap.min.css'

function Login() {
    const [values, setValues] = useState({
        email: '',
        password: ''
    });

    const navigate = useNavigate();

    const [errors, setErrors] = useState({});

    const handleInput = (event) => {
        setValues(prev => ({ ...prev, [event.target.name]: event.target.value }));
    }

    const handleSubmit = (event) => {
        event.preventDefault();
        setErrors(validation(values));

        if (errors.email === "" && errors.password === "") {
            axios.post('http://localhost:3030/login/', {
                email: values.email,
                password: values.password
            })
                .then(res => {
                    console.log("Response from server:", res.data); // Debugging statement
                    if (res.data === "Success") {
                        console.log("Login successful"); // Debugging statement
                        navigate('/Dashboard'); // go to the home page
                    } else {
                        console.log("Login failed"); // Debugging statement
                        alert("Email and password do not match.");
                    }
                })
                .catch(err => {
                    console.error("Login Error from Frontend:", err);
                    alert("An error occurred during login. Please try again later.");
                });
        }
    }

    return (

        <div id='fake_root'>      
            <div className='back_to_home'>   
                <Link to='/Home' className=''> &lt;-- Home Page </Link>
            </div>

        <div className='the_outest_div_box'>

            <div className='d-flex justify-content-center align-items-center bg-primary min-vh-100'>
                {/* <div className='bg-white p-4 rounded shadow' style={{ width: '350px' }}> */}
            
                <div className='signin_main'>              
                    <div className='signin_form_main'>
                    <h2> Login* </h2>

                    <form onSubmit={handleSubmit}>
                        <div className='mb-3 form-3'>
                            <label htmlFor='email'> <strong> Email </strong>  </label>
                            <input type='email' placeholder='Enter Email' name='email' id="email" autoComplete="email"
                                onChange={handleInput} className=''
                            />
                            {errors.email && <span className='text-danger'> {errors.email} </span>}
                        </div>

                        <div className='mb-3 form-3'>
                            <label htmlFor='password'> <strong> Password </strong> </label>
                            <input type='password' placeholder='Enter password' name='password' id="password" autoComplete="new-password"
                                onChange={handleInput} className=''
                            />
                            {errors.password && <span className='text-danger'> {errors.password} </span>}
                        </div>

                        <p> You are agree to our terms and policies </p>
                        <button type='submit' className='btn btn-success'>  Log in  </button>

                        <div className='form_login'>   
                            <a href='/signup' target='_blank' className='btn btn-default border'>  Create Account </a>
                        </div>

                    </form>
                {/* </div> */}
                </div>
                </div>
            </div>
       </div>
        // *the outest div box
    </div>
    // outest than outest
    );
}

export default Login;
