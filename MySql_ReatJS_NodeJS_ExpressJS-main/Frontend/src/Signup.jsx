import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import validation from './SignupValidation';
import axios from 'axios';
import './App.css'
import './signinupStyle.css'


function Signup() {
    const [values, setValues] = useState({
        name: '',
        email: '',
        password: ''
    });

    // natigation part 1
    //const navigate = useNavigate();

    const [errors, setErrors] = useState({});

    const handleInput = (event) => {
        setValues(prev => ({ ...prev, [event.target.name]: event.target.value }));
    };

    const handleSubmit = (event) => {
        event.preventDefault();

        const validationErrors = validation(values);
        setErrors(validationErrors);

        // Check if there are no validation errors
        if (Object.values(validationErrors).every(error => error === "")) {
            // Omit the 'id' field from the 'values' object
            const { id, ...dataWithoutId } = values;
            axios.post('http://localhost:3030/signup/', dataWithoutId)
                .then(res => {
                    console.log("Signup info submitted successfully. ", res);
                    // Redirect the user to the home page, natigation part-2 
                    //navigate('/');
                    window.location.href = '/Home';
                })
                .catch(err => console.log("Signup Error in Frontend. ", err));
        }
    };

    return (
        <div id="fake_root">      
            <div className='back_to_home'>   
                <Link to='/Home' className=''> &lt;-- Home Page </Link>
            </div>

        <div className='the_outest_div_box'>  
            <div className='d-flex justify-content-center align-items-center bg-primary vh-100'>

                {/* <div className='bg-white p-3 rounded w-25'> */}
                    <div className='signin_main'>  
                    <div className='signin_form_main'>
                    <h2> Register </h2>

                    <form action='' onSubmit={handleSubmit}>
                        <div className='mb-3 form-3'>
                            <label htmlFor='name'> <strong> Name </strong>  </label>
                            <input type='text' placeholder='Enter Name' name='name' id="name" autoComplete="name"
                                onChange={handleInput} className=''
                            />
                            {errors.name && <span className='text-danger'> {errors.name} </span>}
                        </div>

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

                        <p> You agree to our terms and policies 
                            **
                        </p>

                        <button type='submit' className='btn btn-success'>  Register  </button>
                        <div className='form_login'>   
                            <Link to='/Login' className='btn btn-default border'>  Login </Link>
                        </div>

                    </form>
                    </div>
                {/* </div> */}
                </div>
            </div>
        </div>
        // *the outest div box
    </div>
    //outest than outest
    );
}

export default Signup;
