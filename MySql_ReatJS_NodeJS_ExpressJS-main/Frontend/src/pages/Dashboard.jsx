import React from 'react'
//import '../App.css';
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import './DashboardMessage.css'; // Import CSS file for styling

import ChatBot from '../chatBot/ChatBot';

function Dashboard() {
  return (
    <Sidebar> 
        <div>
          {/* <Header /> */}
          <div className='container_dash'>
        
      
            <h2>Welcome to Auto Proof Generation Software</h2>

            <p>Generate accurate and reliable proofs effortlessly with our advanced 
              auto proof generation software. Say goodbye to manual proofreading and hello to efficiency!</p>


              <h3>    Custom T-Shirts! </h3>
           
              <p> At ABCD Company , we're passionate about transforming your ideas into wearable outfits.
                With a commitment to quality, creativity, and customer satisfaction, we specialize 
                in crafting custom t-shirts that reflect your unique style and personality.

              </p>
              <h3> Our Story: </h3>
              <p>
                Founded 10 years ago, ABCD Company started as a small venture with a big vision: 
                to revolutionize the way people express themselves through clothing. Since then, 
                we've grown into a reputable brand known for our exceptional craftsmanship and innovative designs.
              </p>

        </div>
    </div>
  </Sidebar>


    
  )
}

export default Dashboard