import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './FileUploadStyle.css'
import Sidebar from '../components/Sidebar'

function FileUpload() {
    const [data, setData] = useState(null); 
    const [style, setStyle] = useState('');
    const [color, setColor] = useState('');
    const [desc, setDesc] = useState('');
    const [price, setPrice] = useState('');
    const [file, setFile] = useState(null);

    const handleFile = (e) => {
        setFile(e.target.files[0]);
    };

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = () => {
        axios.get('http://localhost:3030/lastpost')
            .then(res => {
                if (res.data) {
                    setData(res.data);
                    setStyle(res.data.style);
                    setColor(res.data.color);
                    setDesc(res.data.desc);
                    setPrice(res.data.price);
                }
            })
            .catch(err => console.log(err));
    };

    const handleUpload = () => {
        const formData = new FormData();
        formData.append('style', style);
        formData.append('color', color);
        formData.append('desc', desc);
        formData.append('price', price);
        formData.append('image', file);

        axios.post('http://localhost:3030/upload/', formData)
            .then(res => {
                if (res.data.Status === "Success") {
                    console.log("Upload successful");
                    fetchData();
                } else {
                    console.log("Upload failed");
                }
            })
            .catch(err => console.log("Error during upload: ", err));
    };
    

    // delete:
  //   const handleDelete = () => {
  //     axios.delete('http://localhost:8081/delete/' + data.id) // Adjust the endpoint as per your backend implementation
  //         .then(res => {
  //             console.log("Item deleted successfully");
  //             fetchData(); // Refresh data after deletion
  //         })
  //         .catch(err => console.log("Error during delete: ", err));
  // };

  

    return (
        <Sidebar> 
        <div className='tshirt_container'>
          {/* Analytics page */}
          
          <div className='input_container'> 
            {/* Input fields to set style, color, description, and price */}
            <form  className='form-contro_big'>  
            
            <h2 style={{ margin: '30px 0'}}> Adding a new Tshirt </h2>
            <div className="add_order_element">
            <label htmlFor="style">Style: </label>
            <input type='text' placeholder='Style' value={style} onChange={(e) => setStyle(e.target.value)} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />
            </div>

            <div className="add_order_element">
                <label htmlFor="color">Color: </label>
                <input type='text' placeholder='Color' value={color} onChange={(e) => setColor(e.target.value)} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
                <br />
            </div>

            <div className="add_order_element">
            <label htmlFor="description">Description: </label>
            <input type='text' placeholder='Description' value={desc} onChange={(e) => setDesc(e.target.value)} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />
            </div>

            <div className="add_order_element">
            <label htmlFor="price">Price: </label>
            <input type='text' placeholder='Price' value={price} onChange={(e) => setPrice(e.target.value)} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />
            </div>

            {/* File input for image */}
            <div className="add_order_element" > 
            <label htmlFor="imageInput" >Choose Image:</label>
            <input id="imageInput" type='file' onChange={handleFile} style={{ marginBottom: "30px", marginTop: '20px'}}/> 
            {/* Button to upload */}
            </div>
       
            <button onClick={handleUpload} className='btn-submit'> Submit </button>  <br/>
            {/* Display uploaded data */}
            </form>

          </div>

          <div className='display_container'> 
              {data && (
                  <div>
                    <br/>
                    {/* Display image if available */}
                    {data.image && <img src={`http://localhost:3030/images/${data.image}`} alt="Uploaded" style={{ width: "350px", height: "300px" }} />}
                      
                      <br/> <br/>
                      <p><h4> Style: {data.style} </h4></p>    

                      <p><h4> Color: {data.color} </h4></p>            

                      <p><h4> Description: {data.description} </h4></p>

                      <p><h4> Price: {data.price} </h4></p>

                      {/* <button >Edit</button>
                      <button >Delete</button> */}
                  </div>  

                  
              )}

          </div> 

        </div>
        </Sidebar>
    );
}

export default FileUpload;








