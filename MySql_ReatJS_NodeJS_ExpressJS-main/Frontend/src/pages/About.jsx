import React from 'react'
import { useState } from 'react';
import axios from 'axios';
import './FileUploadStyle.css'
import './about.css'
import Sidebar from '../components/Sidebar';


function About() {
  const [ordernumber, setOrderNumber] = useState('');
  const [order, setOrder] = useState(null);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState('middle'); // State variable to track the selected image size
  const [imageCorner, setImageCorner] = useState('center'); // State variable to move image to the corner
  const colorsLib = {
    "PMS123": "#FDB927",
    "PMS268": "#552583",
    "Black": "#000000",
    "PMS5605C": "#203731",
    "PMS1235C": "#FFB612",
    "GOLD872C": "#ffd700",
    "PMS872C": "#ffd700",
    "PMS187C": "#EF3340",
  
    "PMS202C": "#98002e",
    "PMS137C": "#f9a01b",
    "PMS123C": "#ffc72c",
    "PMSBlackC": "#000000",
    "PMS293C": "#006bb6",
    "PMS199C": "#ed174c",
    "PMS289C": "#002b5c",
    "PMS877C": "#c4ced4",
    "PMS268C": "#5a2b81",
    "PMS431C": "#63727a",
    "PMS282C": "#002d62",
    "PMSCoolGray5C": "#bec0c2",
    "PMS186C": "#c8102e",
    "PMS7687C": "#1d428a",
    "PMS356C": "#008348",
    "PMS874C": "#bb9753",
    "PMS174C": "#a73832",
    "PMS472C": "#fab383",
    "PMS2388C": "#0053bc",
    "yellow": "#ffc72c",
    "YELLOW": "#ffc72c",
    "PURPLE": "#552583",
    "white": "#FFFFFF",
    "WHITE": "#FFFFFF",
    "BLACK": "#000000"
  };

  

  // Function to handle fetching order details
  const fetchOrderDetails = () => {
      axios.get(`http://localhost:3030/allone/${ordernumber}/`)
          .then(response => {
              setOrder(response.data);
              setError(null);
          })
          .catch(error => {
              console.error('Error fetching order details:', error);
              setError('Error fetching order details. Please try again.');
          });
  };

      // Formating data to "YYYY-MM-DD" format
      function formatDate(dateString) {
        const date = new Date(dateString);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
    

    const displayOrderDetails = () => {
      if (order) {
          return (
              <div>
                  {/* <h2>Order Details</h2> */}
                  <ul>
                      {/* {Object.entries(order).map(([key, value]) => (
                          <li key={key}><strong>{key}:</strong> {value}</li>
                      ))} */}
                  </ul>
              </div>
          );
      } else {
          return null;
      }
  };

  // Function to create a color rectangle based on a color key
function colorRec(colorKey) {
  const color = colorsLib[colorKey]; // Retrieve the color code from colorsLib using the color key
  if (color) { // Check if the color code exists
    return (
      <div className='color-block' style={{ backgroundColor: color, color: 'white' }}>
        {colorKey} 
        {/* // Display the color key or name */}
      </div>
    );
  } else {
    return null; 
    // giving null if the color code is not found
  }
}

function ColorBlocks({ order }) {
  // Extract color properties dynamically based on the keys that might be present in the order
  const colorKeys = Object.keys(order).filter(key => key.startsWith('c') && order[key] != null);

  return (
    <div>
      {colorKeys.map(key => colorRec(order[key]))} 
      {/* // Passing the color name or key from the order object */}
    </div>
  );
}


const handleSizeChange = (size) => {
  setImageSize(size);
};

const handleCornerChange = (corner) => {
  setImageCorner(corner);
};



const getImageContainerStyles = () => {
  if (imageCorner === 'center') {
    return {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "380px",
    };
  } else if (imageCorner === 'corner') {
    return {
      display: "flex",
      justifyContent: "flex-start",
      alignItems: "center",
      height: "30px",
      marginLeft: "280px",
      marginTop: "180px",
      border: "none",
    };
  }
};


const getLogoContainerStyles = () => {
  if (imageSize === 'big'&& imageCorner === 'center' ) {
    return {
      position: "absolute",
      marginLeft: "115px",
      marginTop: "-270px",
      border: "none !important",

    };
  } else if (imageSize === 'small'&& imageCorner === 'center') {
    return {
      position: "absolute",
      marginLeft: "145px",
      marginTop: "-230px",
      border: "none !important",
    };
  }else if (imageSize === 'small'&& imageCorner === 'corner') {
    return {
      position: "absolute",
      marginLeft: "185px",
      marginTop: "-240px",
      border: "none !important",
    };
  }else if (imageSize === 'middle'&& imageCorner === 'corner') {
    alert("This size not recommend.");
    return {
      position: "absolute",
      marginLeft: "175px",
      marginTop: "-305px",
      border: "none !important",
    };
  }else if (imageSize === 'big'&& imageCorner === 'corner') {
    alert("This size not work.");
    return {
      position: "absolute",
      marginLeft: "185px",
      marginTop: "-315px",
      border: "none !important",
    };
  }
  
};



  return (
    <Sidebar>
      <div>
          
       {/*  ****************************** enter order number ******************************  */}
        <div className='proof_generator_page_main'>  
            
            {/* <div>  <h2> Automatic Virtual Proof Generator  </h2>     </div> */}

            <div className="proof_generator_content_wrapper">
            
          <div className='proof_generator_page_inputnum'>
              <label htmlFor="ordernumber">Enter Order Number:</label>
              <input type="text" id="ordernumber" value={ordernumber} onChange={(e) => setOrderNumber(e.target.value)} />
              
              <div className='detial_button'> 
                <button onClick={fetchOrderDetails}> Details </button>
              </div>
              <p> ** Generate Order Details ** </p>
            

              <div className='proof_generator_page_position'>
                <input
                  type="radio"
                  id="center"
                  name="corner"
                  value="center"
                  checked={imageCorner === 'center'}
                  onChange={() => handleCornerChange('center')}
                />
                <label htmlFor="center">Center</label>
                <input
                  type="radio"
                  id="corner"
                  name="corner"
                  value="corner"
                  checked={imageCorner === 'corner'}
                  onChange={() => handleCornerChange('corner')}
                />
                <label htmlFor="corner">Corner</label>
              </div>

              <div className='proof_generator_page_size'> 
                <input
                  type="radio"
                  id="small"
                  name="size"
                  value="small"
                  checked={imageSize === 'small'}
                  onChange={() => handleSizeChange('small')}
                />
                <label htmlFor="small"> Small </label>
                <input
                  type="radio"
                  id="middle"
                  name="size"
                  value="middle"
                  checked={imageSize === 'middle'}
                  onChange={() => handleSizeChange('middle')}
                />
                <label htmlFor="middle"> Middle </label>
                <input
                  type="radio"
                  id="big"
                  name="size"
                  value="big"
                  checked={imageSize === 'big'}
                  onChange={() => handleSizeChange('big')}
                />
                <label htmlFor="big"> Large </label>
            </div>

            {error && <p>{error}</p>}
          </div>

        
         
              {/* // ****************************   formating session ****************************  */}

              <div className='display_container'> 
                {order && (
                    <div>

                        <div className="form-container">
                    <div className="top-part">
                      <div className="screen-info">
                        <div className="screen-block">

                          {/* ************************  FRONT SCREENS left side color areas **********************  */}
                        <div className="screen-block"> 
                          <strong>PRINT: {order.tname} - {order.logodesc} </strong>
                          <hr></hr>

                        </div>
                          <div className="screen-block">
                            <strong> FRONT SCREENS </strong>
                            <hr></hr>

                          </div>
                                <div  className="color-rec">
                                {ColorBlocks({ order })}                     
                                </div>
                        </div>


                        {/* ******************************* BACK SCREENS right side color areas *******************************  */}
                        <div>
                          <div className="screen-block">
                              <strong>Graphic#: {order.id}</strong>
                              <hr></hr>
                          </div>

                          <div className="screen-block">
                            <strong> BACK SCREENS </strong>
                            <hr></hr>
                          </div>

                          <div>
                              <div className="color-block-back" >
                                      
                              </div>
                          </div>


                        </div>
                      </div>

                      {/* // ******************************* MIDDLE AREA ******************************* */}

                  <div className="mid-session">   
                        <div className="input-field">
                          <label htmlFor="customerOrder"> <strong > Customer Order #: </strong> </label>
                          <p>{order.ord}</p>
                        </div>

                        <div className="input-field">
                          <label htmlFor="customerName"> <strong> Customer Name: </strong></label>
                          <p>{order.customer}</p>
                        </div>

                        <div className="input-field">
                          <label htmlFor="orderDate">  <strong> Date: </strong></label>
                          <p>{formatDate(order.start)}</p>
                        </div>

                      </div>  
                      {/* mid-session end **************************** */}

                    </div>

                    {/* //******************************* TABLE AREA ******************************* */}
                    <table id="orderTable">
                      <thead>
                        <tr>
                          {/* <th>T-shirt Image Description</th> */}
                          <th>Order # </th>
                          <th>Style # </th>
                          <th>Color</th>
                          <th>Team</th>
                          <th>Logo Desc</th>
                          <th>XS</th>
                          <th>S</th>
                          <th>M</th>
                          <th>L</th>
                          <th>XL</th>
                          <th>2XL</th>
                          <th>3XL</th>
                          <th>Total</th>
                          <th>Start</th>
                          <th>End</th>
                      
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          {/* <td>{order.bdesc}</td> */}
                          <td>{order.ord}</td>
                          <td>{order.style}</td>
                          <td>{order.color}</td>
                          <td>{order.tname}</td>
                          <td>{order.logodesc}</td>
                          <td>{order.xs}</td>
                          <td>{order.s}</td>
                          <td>{order.m}</td>
                          <td>{order.l}</td>
                          <td>{order.xl}</td>
                          <td>{order.xxl}</td>
                          <td>{order.xxxl}</td>
                          <td>{order.ttl}</td>
                          <td>{formatDate(order.start)}</td>
                          <td>{formatDate(order.end)}</td>
              
                        </tr>
                      </tbody>
                    </table>

                    {/* //******************************* SMALL TSHIRT IMAGE AREA ******************************* */}
                    <div className="image-container" id="imageContainer">
                          {/* Placeholder for additional display */}
                          <div > {order.image && <img src={`http://localhost:3030/images/${order.image}`} className="logo-container" alt="logo" style={{ width: "100px", height: "110px" }} />} </div>
                          <div className="desc_containe" > {order.description} </div>
                        </div>


                        <div id ="myContainer">
                        {order.image && <img src={`http://localhost:3030/images/${order.image}`} className="tshirt-container" alt="Uploaded" style={{ width: "350px", height: "380px" }} />}
                      
                              {/* <div className="container" style={getImageContainerStyles()}> */}
                              {/* <div id="myAnimation" style={getImageContainerStyles()}> */}
                              <div id="myAnimation" style={getLogoContainerStyles()}>
                                {order && (
                                  <img
                                    src={`http://localhost:3030/images/${order.imagel}`}
                                    className="tshirt-container"
                                    alt="Uploaded"
                                    style={{
                                      width: imageSize === 'big' ? "120px" : imageSize === 'middle' ? "90px" : "70px",
                                      height: imageSize === 'big' ? "120px" : imageSize === 'middle' ? "90px" : "30px",
                                      objectFit: "contain"
                                    }}
                                  />
                                )}
                              {/* </div> */}
                            </div>  
                        </div>

                      </div>
                      </div>  
                  )}

              </div> 
              {/* display_container_end */}
              {/* ATesing block here END */}

            {error && <p>{error}</p>}
            {displayOrderDetails()}
         </div>

         </div>


{/* // ****************************   formating session ****************************  */}

   </div>   
   </Sidebar>

  );
}

export default About



